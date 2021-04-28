from python:3.8.5
run rm -rf /var/lib/apt/lists/*
run apt-get install apt-transport-https && apt-get update
env DEBIAN_FRONTEND=noninteractive
run apt-get install -y libglib2.0-dev libsm6 libxrender1 libxext-dev vim software-properties-common curl build-essential libgl1 libopencv-dev libvulkan-dev
run apt-get install -y tzdata
run ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && dpkg-reconfigure -f noninteractive tzdata
run python -m pip install -U pip -i https://mirrors.cloud.tencent.com/pypi/simple
run python -m pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple
workdir /deploy
copy requirements.txt requirements.txt
run python -m pip install -r requirements.txt
run python -m pip install tritonclient==2.8.0 -i https://pypi.ngc.nvidia.com --trusted-host pypi.ngc.nvidia.com
copy . .
ENV PYTHONPATH "${PYTHONPATH}:/deploy"
