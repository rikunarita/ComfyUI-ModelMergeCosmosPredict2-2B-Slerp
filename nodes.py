import torch
import comfy.model_management
import comfy.model_patcher
import comfy_extras.nodes_model_merging
import math
import re

# ──────────────────────────────────────────────────────────────
# Slerp（球面線形補間）実装 - 数値的安定性を最優先
# ──────────────────────────────────────────────────────────────
def slerp_safe(v0, v1, t, DOT_THRESHOLD=0.9995, EPS=1e-7):
    """
    数値的に安定したSlerp実装
    
    Args:
        v0, v1: torch.Tensor (同じ形状であること)
        t: float (0.0〜1.0)
        DOT_THRESHOLD: dot productがこの値以上ならLinearにフォールバック
        EPS: ゼロ除算防止用の微小値
    
    Returns:
        torch.Tensor: Slerp補間結果
    """
    # 入力をfloat32にキャスト（精度確保）
    v0 = v0.to(torch.float32)
    v1 = v1.to(torch.float32)
    
    # テンソルを1次元にフラット化
    v0_flat = v0.flatten()
    v1_flat = v1.flatten()
    
    # L2正規化
    norm0 = torch.norm(v0_flat)
    norm1 = torch.norm(v1_flat)
    
    # ゼロノーム防止
    if norm0 < EPS or norm1 < EPS:
        # どちらかがゼロに近い場合はLinear補間
        return (v0 * (1.0 - t) + v1 * t).to(v0.dtype)
    
    v0_norm = v0_flat / norm0
    v1_norm = v1_flat / norm1
    
    # 内積計算（cosθ）
    dot = torch.dot(v0_norm, v1_norm)
    dot = torch.clamp(dot, -1.0, 1.0)  # 数値誤差による範囲逸脱防止
    
    # θ = acos(dot)
    theta = torch.acos(dot)
    
    # thetaが非常に小さい場合（ベクトルがほぼ同じ方向）はLinear補間
    if abs(theta) < EPS or dot > DOT_THRESHOLD:
        result_flat = v0_flat * (1.0 - t) + v1_flat * t
    else:
        # Slerp計算式
        # result = (sin((1-t)*θ) / sin(θ)) * v0 + (sin(t*θ) / sin(θ)) * v1
        sin_theta = torch.sin(theta)
        
        # sin_thetaがゼロに近い場合の保護
        if abs(sin_theta) < EPS:
            result_flat = v0_flat * (1.0 - t) + v1_flat * t
        else:
            coef0 = torch.sin((1.0 - t) * theta) / sin_theta
            coef1 = torch.sin(t * theta) / sin_theta
            result_flat = coef0 * v0_norm + coef1 * v1_norm
        
        # 【修正】元のノームをtの比率に応じて線形補間
        interpolated_norm = norm0 * (1.0 - t) + norm1 * t
        result_flat = result_flat * interpolated_norm
    
    # 元の形状に復元
    result = result_flat.reshape_as(v0)
    
    # 元のdtypeに戻す
    return result.to(v0.dtype)


class ModelMergeCosmosPredict2_2B_Slerp(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    """
    Cosmos Predict 2B専用マージノード（Slerp対応）
    各ブロックごとにLinearまたはSlerpを選択可能
    """
    
    CATEGORY = "model/merging/model specific"
    
    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {
            "model1": ("MODEL",),
            "model2": ("MODEL",),
            "merge_mode": (["linear", "slerp"], {"default": "slerp"}),
        }
        
        # 各ブロック用のマージ比率スライダー
        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
        
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
    FUNCTION = "merge_models"

    def merge_models(self, model1, model2, merge_mode, **kwargs):
        # モデルのstate_dictを取得
        sd1 = model1.model.state_dict()
        sd2 = model2.model.state_dict()
        
        # マージ結果を格納する辞書
        merged_sd = {}
        
        # 共通キーと個別キーを分離
        common_keys = set(sd1.keys()) & set(sd2.keys())
        only_in_sd1 = set(sd1.keys()) - common_keys
        only_in_sd2 = set(sd2.keys()) - common_keys
        
        # 各キーに対してマージ処理
        for key in common_keys:
            # キーに対応する比率を取得
            ratio = 1.0  # デフォルト
            
            # キーのプレフィックスを検出して対応する比率を取得
            matched = False
            
            # 固定プレフィックスのチェック
            for prefix in ["pos_embedder.", "x_embedder.", "t_embedder.", 
                          "t_embedding_norm.", "final_layer."]:
                if key.startswith(prefix):
                    ratio = kwargs.get(prefix, 1.0)
                    matched = True
                    break
            
            # blocks.N.のパターンを検出
            if not matched:
                # 【修正】re.match -> re.search に変更し、キー内のどこかに含まれていれば検出
                match = re.search(r'blocks\.(\d+)\.', key)
                if match:
                    block_num = int(match.group(1))
                    block_key = "blocks.{}.".format(block_num)
                    ratio = kwargs.get(block_key, 1.0)
                    matched = True
            
            # マージモードに応じた計算
            if merge_mode == "slerp" and matched:
                # Slerp適用
                try:
                    # テンソルの形状が一致していることを確認
                    if sd1[key].shape == sd2[key].shape:
                        merged_sd[key] = slerp_safe(sd1[key], sd2[key], ratio)
                    else:
                        # 形状が異なる場合はLinearフォールバック
                        merged_sd[key] = sd1[key] * (1.0 - ratio) + sd2[key] * ratio
                except Exception as e:
                    # 【修正】エラーメッセージの改善
                    print(f"[Cosmos Slerp] Warning: Slerp failed for {key}, falling back to linear. Error: {str(e)}")
                    # フォールバック：Linear補間
                    merged_sd[key] = sd1[key] * (1.0 - ratio) + sd2[key] * ratio
            else:
                # Linear補間（加重平均）
                merged_sd[key] = sd1[key] * (1.0 - ratio) + sd2[key] * ratio
        
        # model1にのみ存在するキーはそのままコピー
        for key in only_in_sd1:
            merged_sd[key] = sd1[key].clone()
        
        # model2にのみ存在するキーはコピー（必要に応じて）
        # 通常はmodel1をベースにするため、これはオプション
        # for key in only_in_sd2:
        #     merged_sd[key] = sd2[key].clone()
        
        # 新しいモデルパッチャーを作成
        new_model = model1.clone()
        
        # マージしたstate_dictを読み込む
        # strict=Falseで、余分なキーや不足キーを許容
        try:
            missing_keys, unexpected_keys = new_model.model.load_state_dict(
                merged_sd, 
                strict=False
            )
            
            if missing_keys and len(missing_keys) > 0:
                print(f"[Cosmos Slerp] Warning: {len(missing_keys)} keys not loaded (missing in merged state_dict)")
            if unexpected_keys and len(unexpected_keys) > 0:
                print(f"[Cosmos Slerp] Warning: {len(unexpected_keys)} unexpected keys in merged state_dict")
                
        except Exception as e:
            print(f"[Cosmos Slerp] Error loading state dict: {str(e)}")
            raise
        
        # 【修正】VRAM管理：不要なモデルをアンロード（keep_clone_weights_loaded=Trueを削除）
        comfy.model_management.cleanup_models()
        
        return (new_model,)


NODE_CLASS_MAPPINGS_ADDITIONAL = {
    "ModelMergeCosmosPredict2_2B_Slerp": ModelMergeCosmosPredict2_2B_Slerp,
}

NODE_DISPLAY_NAME_MAPPINGS_ADDITIONAL = {
    "ModelMergeCosmosPredict2_2B_Slerp": "Model Merge Cosmos Predict 2 2B (Slerp)",
}
