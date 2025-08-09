import re, json
import torch

# Turns a dictionary into a class
class Dict2Class(object):      
    def __init__(self, my_dict):          
        for key in my_dict:
            setattr(self, key, my_dict[key])

def get_gpu_name():
    # https://stackoverflow.com/a/48152675
    device_names = torch.cuda.get_device_name()
    device_counts = torch.cuda.device_count()
    print(f"Running {device_counts} GPUs of type {device_names}")

def get_model_mem_footprint(model):
    try: 
        print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB")
    except:
        print("Not able to get the memory footprint of the model")

def get_gpu_memory(model=None):
    # https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    free_reserved = r-a  # free inside reserved
    f = t-a  
    print(f"---------------------------MERMORY USAGE--------------------------------------\n")
    now = datetime.datetime.now()
    print("Current date and time : ", now.strftime("%Y-%m-%d %H:%M:%S"))
    get_gpu_name()
    print(f"""Total memory: {t/1024.0/1024.0:.2f}MB, Reserved: {r/1024.0/1024.0:.2f}MB, 
          Allocated: {a/1024.0/1024.0:.2f}MB, Free: {f/1024.0/1024.0:.2f}MB""")
    print(f"------------------------------------------------------------------------------\n")
    if model:
        get_model_mem_footprint(model)

def load_all_prompts(file_path: str = None) -> str:
    """
    Loads the api key from json file path

    :param file_path:
    :return:
    """
    cur_dir = pathlib.Path(__file__).parent.resolve()
    # Load prompts from file
    if not file_path:
        # Default file path
        file_path = f"{cur_dir}/resources/prompts.json"
    prompts = load_json(file_path)

    return prompts

def postprocess_output(input_string):
    # Remove extra spaces
    output_string = re.sub(r'\s+', ' ', input_string)
    # Remove extra newline characters
    output_string = re.sub(r'\n+', '\n', output_string)
    # Remove leading and trailing whitespace
    output_string = output_string.strip()
    return output_string

def write_excel_df(
    df_list: List,
    sheet_name_list: List,
    writer: pd.ExcelWriter = None,
    close_writer: bool = False,
    save_file_path: str = None,
    append_mode: bool = False,
):
    """
    Save a list of df in different sheets in one excel file.
    Args:
        writer:
        df_list:
        sheet_name_list:
        close_writer:
        save_file_path:
        append_mode:

    https://stackoverflow.com/questions/20219254/how-to-write-to-an-existing-excel-file-without \
    -overwriting-data-using-pandas
    https://www.geeksforgeeks.org/how-to-write-pandas-dataframes-to-multiple-excel-sheets/

    Returns:
    """
    if save_file_path:
        if append_mode:
            writer = pd.ExcelWriter(save_file_path, mode="a", engine="xlsxwriter")
        else:
            writer = pd.ExcelWriter(save_file_path, engine="xlsxwriter")
    # Write each dataframe to a different worksheet
    assert len(df_list) == len(sheet_name_list)
    for index in range(len(df_list)):
        df_list[index].to_excel(writer, sheet_name=sheet_name_list[index])
    # Close the Pandas Excel writer and output the Excel file.
    if close_writer:
        writer.close()
    return

def load_api_key(file_path: str = None) -> str:
    """
    Loads the api key from json file path

    :param file_path:
    :return:
    """
    cur_dir = pathlib.Path(__file__).parent.resolve()
    # Load config values
    if not file_path:
        # Default file path
        file_path = f"{cur_dir}/resources/config.json"
    config = load_json(file_path)
    api_key = config["OPENAI_API_KEY"]

    return api_key

def write_to_file(path: str, input_text: str) -> None:
    """
    This function opens a file and writes the given input
    string onto the file.

    :param path: Path to the file
    :param input_text: The text to be written on the file
    """
    with open(path, "w", encoding="utf-8") as open_file:
        open_file.write(input_text)
    open_file.close()

def load_json(path: str) -> Any:
    """
    This function opens and JSON file path
    and loads in the JSON file.

    :param path: Path to JSON file
    :type path: str
    :return: the loaded JSON file
    :rtype: dict
    """
    with open(path, "r",  encoding="utf-8") as file:
        json_object = json.load(file)
    return json_object

def get_base_name_from_hf_path(hf_path):
    """
    Can be something like: 
    Eg. hf_path:  multi_x_science, shubhamagarwal92/rw_2308_filtered
    """

    base_name = os.path.split(hf_path)[1]
    return base_name
