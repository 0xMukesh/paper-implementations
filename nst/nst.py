from torch import nn
import torch
from torchvision import models, transforms, utils
from PIL import Image


class NST(nn.Module):
    def __init__(self):
        super().__init__()

        self.content_layers = ["21"]  # conv4_2
        self.style_layers = [
            "0",
            "5",
            "10",
            "19",
            "28",
        ]  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        self.model = models.vgg19(weights="DEFAULT").features[:29]  # type: ignore

    def forward(self, x):
        content_features = []
        style_features = []

        for i, layer in enumerate(self.model):
            x = layer(x)

            if str(i) in self.content_layers:
                content_features.append(x)

            if str(i) in self.style_layers:
                style_features.append(x)

        return content_features, style_features


def load_image(path, device, transform):
    img = Image.open(path)
    img = transform(img).unsqueeze(0)  # add batch dimension
    img = img.to(device)
    return img


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

content_img_path = "/content/images/content.jpg"
style_img_path = "/content/images/style.jpg"

content_img = load_image(content_img_path, device, transform)
style_img = load_image(style_img_path, device, transform)
generated_img = content_img.clone().requires_grad_(True)

total_steps = 6000
learning_rate = 0.001
alpha = 1
beta = 0.01

model = NST().to(device).eval()
optimizer = torch.optim.Adam([generated_img], learning_rate)

with torch.no_grad():
    content_features, _ = model(content_img)
    _, style_features = model(style_img)

for step in range(total_steps):
    generated_content_features, generated_style_features = model(generated_img)

    content_loss = torch.tensor(0.0, device=device)
    style_loss = torch.tensor(0.0, device=device)

    for content_feature, gen_content_feature in zip(
        content_features, generated_content_features
    ):
        _, channel = gen_content_feature.shape
        gen_content_feature = gen_content_feature.view(channel, -1)
        content_feature = content_feature.view(channel, -1)

        content_loss += torch.mean((gen_content_feature - content_feature) ** 2)

    for style_feature, gen_style_feature in zip(
        style_features, generated_style_features
    ):
        _, channel, height, width = gen_style_feature.shape
        gen_style_feature = gen_style_feature.view(channel, -1)
        style_feature = style_feature.view(channel, -1)

        G = gen_style_feature @ gen_style_feature.T
        A = style_feature @ style_feature.T
        style_loss += torch.mean((G - A) ** 2)

    total_loss = alpha * content_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(total_loss)
        utils.save_image(generated_img, f"/content/outputs/{step}.png")
