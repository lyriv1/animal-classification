from datasets import Dataset,DatasetDict,Image as Image_from_datasets, load_dataset, load_from_disk


from data_process_utils import data_process

#------------------------------------------数据处理-----------------------------------------------

train_res_json = data_process.img_label_process('./input/train_data',False,'train')
validation_res_json = data_process.img_label_process('./input/validation_data',False,'validation')

data_process.update_json(train_res_json,'./temp/train.json','train')
data_process.update_json(validation_res_json,'./temp/validation.json','validation')

train_dataset = Dataset.from_json('./temp/train.json')
validation_dataset = Dataset.from_json('./temp/validation.json')

dataset = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset
})


dataset = dataset.cast_column("images", Image_from_datasets())
# os.remove("data.hf")


dataset.save_to_disk("./data.hf")