from setuptools import setup, find_packages
import os

def read_readme():
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    return "Depth-Anything-V2: State-of-the-art monocular depth estimation model"

# # 恢复依赖（必须安装这些库才能运行）
# install_requires = [
#     "torch>=2.0.0",
#     "torchvision>=0.15.0",
#     "timm>=0.9.0",
#     "einops>=0.6.0",
#     "numpy>=1.21.0",
#     "Pillow>=9.0.0",
#     "opencv-python>=4.5.0",
#     "tqdm>=4.64.0",
# ]

setup(
    name="depth_anything_v2",
    version="1.0.2",
    author="Depth-Anything Team",
    author_email="contact@depth-anything.com",
    description="Official implementation of Depth-Anything-V2",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/LiheYoung/Depth-Anything-V2",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Computer Vision :: Depth Estimation",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="depth estimation, monocular depth, computer vision",
    
    # 关键：find_packages会自动识别根目录下的depth_anything_v2包
    packages=find_packages(exclude=["tests", "examples", "assets", "build", "output"]),
    
    # 确保depth_anything_v2内的所有代码文件被包含
    include_package_data=True,
    # 若depth_anything_v2内有子目录（比如model），也会被自动包含（只要有__init__.py）
    
    # install_requires=install_requires,
    python_requires=">=3.8, <3.12",
    zip_safe=False,
)
