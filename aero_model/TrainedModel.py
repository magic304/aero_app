from keras.utils import plot_model

from aero_model.PreTrain import *
from aero_model.Transfer import *

# from tensorflow.python.keras.optimizers import
# 确定随机种子，确保每次结果一样
# from tensorflow.python.keras.optimizers import
# 确定随机种子，确保每次结果一样

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

absolute_path = os.path.split(sys.argv[0])[0]
os.environ["PATH"] += os.pathsep + absolute_path + '/Graphviz/bin'  # 注意修改你的路径

class TrainedModel():
    def reshape_input(self, x) -> None:
        x_st = x.reshape((x.shape[0], 1, x.shape[1]))
        return x_st

    def predict(self, df=None):
        self.x_ori = df[self.trained_model.x_name].to_numpy()

        # 归一化
        self.x = self.trained_model.x_scaler.transform(self.x_ori)

        # reshape
        self.x_st = self.reshape_input(self.x)
        self.y_pre = self.trained_model.model.predict(self.x_st)
        self.y = self.trained_model.y_scaler.inverse_transform(self.y_pre)

    # 结果转换为dataframe
    def result_to_df(self):
        df1 = pd.DataFrame(data=self.x_ori, columns=self.trained_model.x_name)
        df2 = pd.DataFrame(data=self.y, columns=self.trained_model.y_name)
        self.df = pd.concat([df1, df2], axis=1)
        print(self.df)
        # 获取summary
        self.summary = self.trained_model.model.summary()

    def model_load_job(self, model_name=None):
        model_name = model_name or '222'
        self.trained_model = self.loadModel(model_name)
        plot_model(self.trained_model.model, to_file='model.png', show_shapes=True, show_layer_activations=True,
                   expand_nested=True, show_dtype=True)

    @staticmethod
    def loadModel(model_name=None):
        if model_name is None:
            obj = None
            return obj
        else:
            model_name = model_name
            path = f"{absolute_path}/dat/model/{model_name}/{model_name}.pkl"
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            from keras.models import load_model
            obj.model = load_model(f'{absolute_path}/dat/model/{model_name}/{model_name}.h5',
                                   custom_objects={"mmd_loss": mmd_loss})
            return obj


if __name__ == '__main__':
    model = TrainedModel()
    model.model_load_job('222')
    print(model.trained_model.x_name)
    tmodel = TransferModel()
    df = tmodel.load_data()
    model.predict(df)
    model.result_to_df()
    print(model.summary)
