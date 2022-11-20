import os
import sys

absolute_path = os.path.split(sys.argv[0])[0]


class TreeData():
    __model = None

    @staticmethod
    def instance() -> 'TreeData':
        if TreeData.__model is None:
            TreeData.__model = TreeData.__loadModel()
        return TreeData.__model

    def __init__(self) -> None:
        self.datas = {'data': [], 'modeling': []}

    def add_node(self, parent_type, name, data_type=None, data_file_name=''):
        node = {'name': name, 'data_type': data_type, 'data_file_name': data_file_name}
        self.datas[parent_type].append(node)
        self.saveModel()

    def remove_node(self, parent_type, inx):
        if (inx < len(self.datas[parent_type])):
            node = self.datas[parent_type][inx]
            data_file_name = node['data_file_name']
            data_type = node['data_type']
            self.__delFile(data_type, data_file_name)
            self.datas[parent_type].pop(inx)  # [inx:inx+1]=[]
            self.saveModel()

    def __delFile(self, data_type, data_file_name):
        path = f"{absolute_path}/dat/{data_type}/{data_file_name}"
        if os.path.exists(path):
            os.remove(path)

    def is_exist(self, parent_type, data_type):
        node_list = self.datas[parent_type]
        for node in node_list:
            if data_type == node['data_type']:
                return True
        return False

    def saveModel(self):
        import pickle
        # file_id=str(uuid.uuid1()) 
        file_id = 'inx_dat'
        if not os.path.exists(f"{absolute_path}/dat/"):
            os.makedirs(f"{absolute_path}/dat/")
        path = f"{absolute_path}/dat/{file_id}.pkl"
        f = open(path, 'wb')
        pickle.dump(self, f)
        f.close()
        return file_id

    @staticmethod
    def __loadModel() -> 'TreeData':
        import pickle
        try:
            file_id = 'inx_dat'
            path = f"{absolute_path}/dat/{file_id}.pkl"
            with open(path, 'rb') as f:
                obj = pickle.load(f)
        except Exception as e:
            obj = TreeData()
        TreeData.model = obj
        return obj


class TreeNode():
    def __init__(self, name, type, data_path, more_info={}) -> None:
        self.name = name
        self.type = type
        self.data_path = data_path
        self.more_info = more_info


if __name__ == "__main__":
    pass
    model = TreeData.loadModel()
    # model.add_node("data","hell1","no_need","no_need")
    # model.remove_node("data",3)

    print(model.datas)
