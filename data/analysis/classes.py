from timeutils import MyDate, MyTime, MyDuration

import pandas as pd

class JSONObject:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, JSONObject(value))
            else:
                setattr(self, key, value)

    def __str__(self):
        return '{'+','.join(f'{key}:{value}' for key, value in self.__dict__.items())+'}'

class Grid:
    def __init__(self, data):
        if isinstance(data, dict):
            self.__load_dict__(data)
        else:
            raise ValueError('The data must be a dictionary.')

    def __load_dict__(self, data):
        self.positions = [JSONObject(v) for _, v in sorted(data['grid'].items(), key=lambda item: int(item[0]))]

    def __str__(self):
        return '{'+'grid:{'+','.join(str(i)+':'+str(p) for i, p in enumerate(self.positions))+'}'+'}'

class Information:
    __supported_versions__ = ['1', '2']
    __header_version__ = 'Version:'
    __participant_name__ = 'Nome_do_sujeito:'
    __session_name__ = 'Nome_da_sessao:'
    __session_result__ = 'Resultado:'
    __grid__ = 'Grade_de_estimulos:'
    __monitor__ = 'Monitor:'
    __start_date__ = 'Data_Inicio:'
    __start_time__ = 'Hora_Inicio:'
    __end_date__ = 'Data_Termino:'
    __end_time__ = 'Hora_Termino:'
    __duration__ = 'Duration:'

    def __init__(self, entry, verbose=True):
        if not entry.endswith('.info.processed'):
            raise ValueError('The file must be a .info.processed file.')
        else:
            if verbose:
                print(f'Loading {entry}...')

        self.__entry__ = entry
        self.__info_file__ = self.__load_info_file__(entry)
        self.version = self.__get_version__()
        self.session_name = self.__get_session_name__()
        self.participant_name = self.__get_participant_name__()
        self.start_date = self.__get_start_date__()
        self.end_date = self.__get_end_date__()
        self.start_time = self.__get_start_time__()
        self.end_time = self.__get_end_time__()
        self.duration = self.__get_duration__()
        self.result = self.__get_result__()
        self.__load_grid__()
        self.__load_monitor__()

    def __load_info_file__(self, entry):
        try:
            info = pd.read_csv(entry, sep='\t', encoding='utf-8', header=None, index_col=0, engine='python')
        except FileNotFoundError:
            print(f'File {entry} not found.')
            return None
        return info

    def __get_session_name__(self):
        return self.__info_file__.loc[self.__session_name__][1]

    def __get_participant_name__(self):
        return self.__info_file__.loc[self.__participant_name__][1]

    def __get_result__(self):
        try:
            result = self.__info_file__.loc[self.__session_result__][1]
        except KeyError:
            return None

        return result

    def __get_start_date__(self):
        return MyDate(self.__info_file__.loc[self.__start_date__][1])

    def __get_start_time__(self):
        return MyTime(self.__info_file__.loc[self.__start_time__][1])

    def __get_end_date__(self):
        try:
            result = MyDate(self.__info_file__.loc[self.__end_date__][1])
        except KeyError:
            return None

        return result

    def __get_end_time__(self):
        try:
            result = MyTime(self.__info_file__.loc[self.__end_time__][1])
        except KeyError:
            return None

        return result

    def __get_duration__(self):
        try:
            result = MyDuration(self.__info_file__.loc[self.__duration__][1])
        except KeyError:
            return None

        return result

    def __get_version__(self):
        try:
            version = self.__info_file__.loc[self.__header_version__][1]
            if version not in self.__supported_versions__:
                print(f'Version {version} is not supported.')
                return None
        except KeyError:
            print(f'File {self.__entry__} has no version information.')
            return None

        return self.__info_file__.loc[self.__header_version__][1]

    def __load_grid__(self):
        json_string = self.__info_file__.loc[self.__grid__][1]
        # add quotes to the json name-value pairs
        json_string = json_string.replace('{', '{"').replace(':', '":').replace(',', ',"')
        # load name-value pairs as corresponding attributes
        self.grid = Grid(eval(json_string))

    def __load_monitor__(self):
        json_string = self.__info_file__.loc[self.__monitor__][1]
        json_string = json_string.replace('{', '{"').replace(':', '":').replace(',', ',"')
        self.monitor = JSONObject(eval(json_string))

    def __str__(self):
        return f'{self.__header_version__}\t{self.version}\n' \
               f'{self.__participant_name__}\t{self.participant_name}\n' \
               f'{self.__session_name__}\t{self.session_name}\n' \
               f'{self.__grid__}\t{self.grid}\n' \
               f'{self.__monitor__}\t{self.monitor}\n' \
               f'{self.__start_date__}\t{self.start_date}\n' \
               f'{self.__start_time__}\t{self.start_time}\n' \
               f'{self.__end_date__}\t{self.end_date}\n' \
               f'{self.__end_time__}\t{self.end_time}\n' \
               f'{self.__duration__}\t{self.duration}\n' \
               f'{self.__session_result__}\t{self.result}'

    def save_to_file(self):
        with open(self.__entry__, 'w', encoding='utf-8') as file:
            file.write(str(self))

    def has_valid_result(self) -> bool:
        valid_results = {
            'Interrompida', 'Concluida',
            'Critério atingido', 'Critério não atingido'}

        return self.result in valid_results

if __name__ == "__main__":
    import os

    from fileutils import cd, list_files

    cd(os.path.join('..', 'estudo1', '12-MED', 'analysis', '2024-03-25'))
    for entry in list_files('.info.processed'):
        info = Information(entry)
        print(info.has_valid_result())