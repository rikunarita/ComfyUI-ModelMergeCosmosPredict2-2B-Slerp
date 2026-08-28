import torch
import math
import comfy.lora
import comfy.model_management

# ──────────────────────────────────────────────────────────────
# Slerp（球面線形補間）実装 - 精度と安定性を最大化
# ──────────────────────────────────────────────────────────────
def slerp_safe(v0, v1, t, DOT_THRESHOLD=0.9995, EPS=1e-7):
    """
    数値的に安定したSlerp実装（符号反転・ノルム補間対応）
    
    Args:
        v0, v1: torch.Tensor (同じ形状であること)
        t: float (0.0〜1.0)
    """
    # 早期リターンによる最適化
    if t == 0.0:
        return v0
    if t == 1.0:
        return v1
        
    original_dtype = v0.dtype
    device = v0.device
    
    # 精度確保のためfloat32にキャスト（v1はv0のデバイスに強制移動）
    v0 = v0.to(torch.float32)
    v1 = v1.to(device=device, dtype=torch.float32)
    
    v0_flat = v0.flatten()
    v1_flat = v1.flatten()
    
    # バイアスやNormなどの1次元テンソル、要素数が少ないテンソルはLinear補間
    if v0.dim() == 1 or v0.numel() < 64:
        result_flat = v0_flat * (1.0 - t) + v1_flat * t
        return result_flat.reshape_as(v0).to(original_dtype)
        
    norm0 = torch.norm(v0_flat)
    norm1 = torch.norm(v1_flat)
    
    # ゼロノーム防止
    if norm0 < EPS or norm1 < EPS:
        result_flat = v0_flat * (1.0 - t) + v1_flat * t
        return result_flat.reshape_as(v0).to(original_dtype)
        
    v0_norm = v0_flat / norm0
    v1_norm = v1_flat / norm1
    
    # 内積計算（cosθ）
    dot = torch.dot(v0_norm, v1_norm)
    dot = torch.clamp(dot, -1.0, 1.0)
    dot_val = dot.item() # GPU-CPU同期はここでのみ行う
    
    # 【重要】符号反転（内積が負の場合、180度以上の回転を避けるためv1を反転）
    if dot_val < 0.0:
        v1_flat = -v1_flat
        v1_norm = -v1_norm
        dot_val = -dot_val
        
    # thetaが非常に小さい場合（ベクトルがほぼ同じ方向）はLinear補間
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
            
        # 【重要】元のノームをtの比率に応じて線形補間
        interpolated_norm = norm0 * (1.0 - t) + norm1 * t
        result_flat = result_flat * interpolated_norm
        
    result = result_flat.reshape_as(v0)
    return result.to(original_dtype)


class ModelMergeCosmosPredict2_2B_Slerp:
    """
    Cosmos Predict 2B専用マージノード（Slerp対応）
    ComfyUIネイティブな add_patches() を使用し、LoRAとの共存とメモリ効率を実現
    """
    
    CATEGORY = "model/merging/model specific"
    
    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {
            "model1": ("MODEL",),
            "model2": ("MODEL",),
            "merge_mode": (["slerp", "linear"], {"default": "slerp"}),
        }
        
        # 各ブロック用のマージ比率スライダー (デフォルトは0.0: model1を維持)
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
        
        # diffusion_model. プレフィックスのパッチ適用済み重みを取得
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
                    
            # blocks.N. のパターンを検出 (正規表現を使わず高速化)
            if not matched and k_unet.startswith("blocks."):
                parts = k_unet.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    block_key = "blocks.{}.".format(parts[1])
                    if block_key in kwargs:
                        ratio = kwargs[block_key]
                        matched = True
                        
            # マッチしないキーや ratio=0.0 の場合は model1 のまま維持
            if not matched or ratio == 0.0:
                continue
                
            # テンソルの取得とパッチ適用
            w1_base = kp1[k][0][0].clone()
            w2_base = kp2[k][0][0].clone()
            
            w1 = comfy.lora.calculate_weight(kp1[k], w1_base, k)
            w2 = comfy.lora.calculate_weight(kp2[k], w2_base, k)
            
            # 形状が異なる場合はスキップ
            if w1.shape != w2.shape:
                continue
                
            # マージ計算
            if merge_mode == "slerp":
                w_new = slerp_safe(w1, w2, t=ratio)
            else:
                w_new = w1 * (1.0 - ratio) + w2 * ratio
                
            # 差分を計算してパッチとして適用
            diff = w_new - w1
            
            # add_patches に {key: (tensor,)} 形式で渡すと "diff" パッチとして認識される
            m.add_patches({k: (diff,)}, 1.0, 1.0)
            
        return (m,)

NODE_CLASS_MAPPINGS = {
    "ModelMergeCosmosPredict2_2B_Slerp": ModelMergeCosmosPredict2_2B_Slerp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelMergeCosmosPredict2_2B_Slerp": "Model Merge Cosmos Predict 2 2B (Slerp)",
}
