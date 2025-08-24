FROM docker.v2.aispeech.com/sjtu/sjtu_wumengyue-zzh_pico:0.0.2
RUN pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ --upgrade pip \
    && pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple/ \
    && apt-get update -y \
    && apt-get install --assume-yes apt-utils \
    && apt-get install git -y \
    && git --version \
    && apt-get install tmux -y \
    && apt-get install gcc libpq-dev -y \
    && pip install git+https://gitee.com/YiweiDD/audioldm_eval \
    && pip install laion-clap \
    && pip install pysptk \
    && pip install pyworld \
    && pip install sed_eval \
    && pip install dcase_util \