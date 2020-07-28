# 作者：极简AI·小宋是呢
# 链接：https://www.zhihu.com/question/326164671/answer/695619342
# 来源：知乎
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

class SpeakerTrainDataset(Dataset):#定义pytorch的训练数据及类
    def __init__(self, samples_per_speaker=1):#每个epoch每个人的语音采样数
        self.dataset = []
        current_sid = -1
        with open(TRAIN_MANIFEST, 'r') as f:
            reader = csv.reader(f)
            for sid, aid, filename, duration, samplerate in reader:#读入manifest，存入二维列表，第一维为说话人，第二维是每个人的若干句话
                if sid != current_sid:
                    self.dataset.append([])
                    current_sid = sid
                self.dataset[-1].append((filename, float(duration), int(samplerate)))
        self.n_classes = len(self.dataset)
        self.samples_per_speaker = samples_per_speaker

    def __len__(self):
        return self.samples_per_speaker * self.n_classes#返回一个epoch的采样数

    def __getitem__(self, sid):#定义采样方式，sid为说话人id
        sid %= self.n_classes
        speaker = self.dataset[sid]
        y = []
        n_samples = 0
        while n_samples < N_SAMPLES:#当采样长度不够时，继续读取
            aid = random.randrange(0, len(speaker))
            audio = speaker[aid]
            t, sr = audio[1], audio[2]
            if t < 1.0:#长度小于1，不使用
                continue
            if n_samples == 0:
                start = int(random.uniform(0, t - 1.0) * sr)
            else:
                start = 0
            stop = int(min(t, max(1.0, start + (N_SAMPLES - n_samples) / SAMPLE_RATE)) * sr)
            _y, _ = load_audio(audio[0], start=start, stop=stop)#读取语音从start 到stop的部分
            if _y is not None:
                y.append(_y)
                n_samples += len(_y)
        return np.array([make_feature(np.hstack(y)[:N_SAMPLES], SAMPLE_RATE).transpose()]), sid#返回特征和说话人id



class ToTensor(object):#转换第二维度和第三维度坐标
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            return torch.FloatTensor(pic.transpose((0, 2, 1)))