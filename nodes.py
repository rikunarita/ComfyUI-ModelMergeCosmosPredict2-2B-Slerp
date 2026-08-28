import torch
import math
import comfy.lora
import comfy.model_management

# ──────────────────────────────────────────────────────────────
# Slerp（球面線形補間）実装
#
# 参考文献:
#   - Model Soups (Wortsman et al., ICML 2022)
#   - TIES-Merging (Yadav et al., NeurIPS 2023) - Sign Elect 概念
#   - MergeKit SLERP (arcee-ai/mergekit)
#   - Shoemake, K. (1985) "Animating Rotation with Quaternion Curves"
#
# 設計方針:
#   - 内積が負の場合に符号反転（最短経路補間）
#   - ノルムは線形補間でスケール維持
#   - 1次元/小規模テンソルはLinearフォールバック
#   - 数値的安定性のためDOT_THRESHOLDでLinearフォールバック
# ──────────────────────────────────────────────────────────────
def slerp_safe(v0, v1, t, DOT_THRESHOLD=0.9995, EPS=1e-6):
    """
    数値的に安定したSlerp実装。

    Args:
        v0: torch.Tensor - model1の重みテンソル
        v1: torch.Tensor - model2の重みテンソル
        t: float - 0.0(model1のみ)〜1.0(model2のみ)
        DOT_THRESHOLD: float - cosθがこの値以上ならLinearフォールバック
        EPS: float - 数値安定化用の微小値

    Returns:
        torch.Tensor - 補間結果（入力と同じdtype・shape）
    """
    # 形状不一致の安全ガード
    if v0.shape != v1.shape:
        return v0 * (1.0 - t) + v1 * t

    # 早期リターン（副作用防止のためclone）
    if t <= 0.0:
        return v0.clone()
    if t >= 1.0:
        return v1.clone()

    original_dtype = v0.dtype
    device = v0.device

    # 精度確保のためfloat32にキャスト（デバイス統一）
    v0 = v0.to(torch.float32)
    v1 = v1.to(device=device, dtype=torch.float32)

    v0_flat = v0.flatten()
    v1_flat = v1.flatten()

    # スカラー・1次元テンソル・小規模テンソルはLinear補間
    # （バイアス、Norm、Embeddingの一部など）
    if v0.dim() <= 1 or v0.numel() < 64:
        result_flat = v0_flat * (1.0 - t) + v1_flat * t
        return result_flat.reshape_as(v0).to(original_dtype)

    norm0 = torch.norm(v0_flat)
    norm1 = torch.norm(v1_flat)

    # ゼロノーム防止
    if norm0 < EPS or norm1 < EPS:
        result_flat = v0_flat * (1.0 - t) + v1_flat * t
        return result_flat.reshape_as(v0).to(original_dtype)

    # L2正規化
    v0_norm = v0_flat / norm0
    v1_norm = v1_flat / norm1

    # 内積計算（cosθ）
    dot = torch.dot(v0_norm, v1_norm)
    dot = torch.clamp(dot, -1.0, 1.0)
    dot_val = dot.item()

    # 【TIES-Merging由来の符号整合】
    # 内積が負 → 180度以上の角度 → 符号反転して最短経路で補間
    if dot_val < 0.0:
        v1_flat = -v1_flat
        v1_norm = -v1_norm
        dot_val = -dot_val

    # ほぼ平行な場合はLinearフォールバック（数値安定性）
    if dot_val > DOT_THRESHOLD:
        result_flat = v0_flat * (1.0 - t) + v1_flat * t
    else:
        theta = math.acos(dot_val)
        sin_theta = math.sin(theta)

        if abs(sin_theta) < EPS:
            result_flat = v0_flat * (1.0 - t) + v1_flat * t
        else:
            coef0 = math.sin((1.0 - t) * theta) / sin_theta
            coef1 = math.sin(t * theta) / sin_theta
            result_flat = coef0 * v0_norm + coef1 * v1_norm

        # ノルムの線形補間（スケール維持）
        interpolated_norm = norm0 * (1.0 - t) + norm1 * t
        result_flat = result_flat * interpolated_norm

    result = result_flat.reshape_as(v0)
    return result.to(original_dtype)


class ModelMergeCosmosPredict2_2B_Slerp:
    """
    Cosmos Predict 2B専用マージノード（Slerp対応）

    ComfyUIネイティブな add_patches() を使用し、
    LoRA・フック・パッチとの完全な共存を実現。
    """

    CATEGORY = "model/merging/model specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {
            "model1": ("MODEL",),
            "model2": ("MODEL",),
            "merge_mode": (["slerp", "linear"], {"default": "slerp"}),
        }

        # デフォルトは0.0（model1を維持）
        argument = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["pos_embedder."] = argument
        arg_dict["x_embedder."] = argument
        arg_dict["t_embedder."] = argument
        arg_dict["t_embedding_norm."] = argument

        # 28個のblocks（0-27）
        for i in range(28):
            arg_dict["blocks.{}.".format(i)] = argument

        arg_dict["final_layer."] = argument

        return {"required": arg_dict}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"

    def merge(self, model1, model2, merge_mode, **kwargs):
        m = model1.clone()

        # diffusion_model. プレフィックスのキーパッチを取得
        kp1 = model1.get_key_patches("diffusion_model.")
        kp2 = model2.get_key_patches("diffusion_model.")

        keys1 = set(kp1.keys())
        keys2 = set(kp2.keys())
        common_keys = keys1 & keys2

        for k in common_keys:
            # "diffusion_model." を剥がしてマッチングに使用
            k_unet = k[len("diffusion_model."):]
            ratio = 0.0
            matched = False

            # 固定プレフィックスのチェック
            for prefix in ["pos_embedder.", "x_embedder.", "t_embedder.",
                           "t_embedding_norm.", "final_layer."]:
                if k_unet.startswith(prefix):
                    ratio = kwargs.get(prefix, 0.0)
                    matched = True
                    break

            # blocks.N. のパターンを検出（正規表現不使用で高速）
            if not matched and k_unet.startswith("blocks."):
                parts = k_unet.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    block_key = "blocks.{}.".format(parts[1])
                    if block_key in kwargs:
                        ratio = kwargs[block_key]
                        matched = True

            # マッチしないキーや ratio=0.0 は model1 のまま維持
            if not matched or ratio == 0.0:
                continue

            # ── model1 の重みを取得（パッチ適用済み）──
            # get_key_patches は [(weight, convert_func), patch1, patch2, ...] を返す
            w1_weight, w1_convert_func = kp1[k][0]
            w1_base = w1_convert_func(w1_weight.clone())
            w1_patches = kp1[k][1:]
            w1 = comfy.lora.calculate_weight(w1_patches, w1_base, k)

            # ── model2 の重みを取得（パッチ適用済み）──
            w2_weight, w2_convert_func = kp2[k][0]
            w2_base = w2_convert_func(w2_weight.clone())
            w2_patches = kp2[k][1:]
            w2 = comfy.lora.calculate_weight(w2_patches, w2_base, k)

            # デバイス・dtypeの統一
            if w1.device != w2.device:
                w2 = w2.to(w1.device)
            if w1.dtype != w2.dtype:
                w2 = w2.to(w1.dtype)

            # 形状不一致はスキップ（安全ガード）
            if w1.shape != w2.shape:
                del w1_base, w2_base, w1, w2
                continue

            # マージ計算
            if merge_mode == "slerp":
                w_new = slerp_safe(w1, w2, t=ratio)
            else:
                w_new = w1 * (1.0 - ratio) + w2 * ratio

            # 差分を計算してパッチとして適用
            # 適用後: weight = w1 + diff = w1 + (w_new - w1) = w_new
            diff = w_new - w1

            # {key: (tensor,)} 形式 → calculate_weight内で "diff" パッチとして認識
            m.add_patches({k: (diff,)}, 1.0, 1.0)

            # 一時テンソルの明示的解放（VRAM節約）
            del w1_base, w2_base, w1, w2, w_new, diff

        return (m,)


NODE_CLASS_MAPPINGS = {
    "ModelMergeCosmosPredict2_2B_Slerp": ModelMergeCosmosPredict2_2B_Slerp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelMergeCosmosPredict2_2B_Slerp": "Model Merge Cosmos Predict 2 2B (Slerp)",
}
