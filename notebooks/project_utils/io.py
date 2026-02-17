import pickle as pkl

def open_pickled_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pkl.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"Error: {e}")