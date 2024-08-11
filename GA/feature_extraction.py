import torch
from torchvision.models import swin_t

# 画像から特徴量を抽出するためにSwin Transformerを使用する
def extract_features(images):
    # Swin Transformerの事前学習済みモデルを使用
    model = swin_t(weights="IMAGENET1K_V1").features
    model.eval()  # モデルを推論モードに設定

    features = []
    for img in images:
        with torch.no_grad():  # 勾配計算を無効化して効率を上げる
            feature = model(img)  # 画像から特徴量を抽出
        features.append(feature)

    return torch.stack(features)
    
    # すべての特徴量をスタックして返す
    return torch.stack(features)