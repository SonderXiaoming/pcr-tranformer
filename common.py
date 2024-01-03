class configurations(object):
    def __init__(self):
        self.batch_size = 200
        self.emb_dim = 256
        self.hid_dim = 512
        self.n_layers = 3
        self.dropout = 0.5
        self.learning_rate = 0.00005
        self.num_steps = 12000  # 總訓練次數
        self.store_steps = 300  # 訓練多少次後須儲存模型
        self.summary_steps = 300  # 訓練多少次後須檢驗是否有overfitting
        self.load_model = True  # 是否需載入模型
        self.store_model_path = "./model"  # 儲存模型的位置
        self.load_model_path = (
            "./model/model_2700"  # 載入模型的位置 e.g. "./ckpt/model_{step}"
        )
        self.attention = False  # 是否使用 Attention Mechanism
        self.train_size_percentage = 0.3


params = configurations()
