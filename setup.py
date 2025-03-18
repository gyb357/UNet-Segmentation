from setuptools import setup, find_packages


setup(
    name='UNet',
    version='0.1.0',
    description='Segmenting Image Regions Using UNet and ResNet Backbone Structures.',
    author='gyb357',
    author_email='gyb357@naver.com',
    url='https://github.com/gyb357/UNet-Segmentation',
    packages=find_packages(exclude=["tests", "docs"]),
    install_requires=[
        'torch',
        'torchvision',
        'tensorboard',
        'matplotlib',
        'pandas',
        'pyyaml',
    ],
    keywords=[
        'computer vision', 'image processing', 'segmentation',
        'unet', 'resnet', 'backbone', 'pytorch', 'torchvision', 'deep learning'
    ],
    python_requires='>=3.6',
    package_data={
        "": ["*.yaml", "*.json"]  # 설정 파일 포함 (필요하면 추가)
    },
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    license="MIT"
)

