import os
import datetime
import torch
import pandas as pd
from random import shuffle
from PIL import Image 
from typing import Tuple, Any
from transformers import CLIPProcessor
from torchvision.transforms import transforms
from warnings import filterwarnings

filterwarnings('ignore', category=FutureWarning, module='transformers')

clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', use_fast=False)

climate_dict = {
    0: 'Tropical, rainforest',
    1: 'Tropical, monsoon',
    2: 'Tropical, savannah',
    3: 'Arid, desert, hot',
    4: 'Arid, desert, cold',
    5: 'Arid, steppe, hot',
    6: 'Arid, steppe, cold',
    7: 'Temperate, dry summer, hot summer',
    8: 'Temperate, dry summer, warm summer',
    9: 'Temperate, dry summer, cold summer',
    10: 'Temperate, dry winter, hot summer',
    11: 'Temperate, dry winter, warm summer',
    12: 'Temperate, dry winter, cold summer',
    13: 'Temperate, no dry season, hot summer',
    14: 'Temperate, no dry season, warm summer',
    15: 'Temperate, no dry season, cold summer',
    16: 'Cold, dry summer, hot summer',
    17: 'Cold, dry summer, warm summer',
    18: 'Cold, dry summer, cold summer',
    19: 'Cold, dry summer, very cold winter',
    20: 'Cold, dry winter, hot summer',
    21: 'Cold, dry winter, warm summer',
    22: 'Cold, dry winter, cold summer',
    23: 'Cold, dry winter, very cold winter',
    24: 'Cold, no dry season, hot summer',
    25: 'Cold, no dry season, warm summer',
    26: 'Cold, no dry season, cold summer',
    27: 'Cold, no dry season, very cold winter',
    28: 'Polar, tundra',
    29: 'Polar, frost'
}

class PretrainDatasetOSVMini(torch.utils.data.Dataset):
    "Pretrain CLIP on osv-mini-129k"
    def __init__(self, split: str, dir: str, shuffle: bool=True, image_size: int=224):
        """Initializes a PretrainDatasetYFCC used for pretraining CLIP.

        Args:
            split (str): dataset split to load.
            dir (str): path to parent directory of the dataset. Must include train_mini.csv, 
                        val_mini.csv, test_images, and train_images directories.
            shuffle (bool, optional): whether the training data should be shuffled. 
                                        Defaults to True.
            image_size (int, optional): the size to which the image should be resized. 
                                        Base uses 224, Large uses 336.
        """
        self.split = split
        self.shuffle = shuffle
        self.image_size = image_size
        self.dir = dir
        self.image_dir = os.path.join(dir, f'{split}_images')
        self.csv_path = os.path.join(dir, f'{split}_mini.csv')

        # basic checks
        assert split in ['train', 'val', 'test']
        assert os.path.exists(dir)
        assert os.path.exists(self.csv_path), f"CSV file does not exist: {self.csv_path}"
        assert os.path.exists(self.image_dir), f"Image directory does not exist: {self.image_dir}"
        
        # load data
        self.df = pd.read_csv(self.csv_path)
        self.df.drop(columns=["creator_username", "creator_id", 'thumb_original_url', 'sequence', "road_index", 'drive_side', 'soil'], inplace=True)
        self.df = self.df.rename(columns={'region': 'state'})
        print(f'Any NaNs: {self.df.isna().any().any()}')

        if shuffle:
            self.df = self.df.sample(frac=1.0, random_state=330)

        shuffle_str = 'shuffled ' if shuffle else ''
        print(f'Initialized {shuffle_str}{split} OSV-Mini-129k dataset with {len(self.df)} samples in metadata.')
        
    def _get_month(self, captured_at: int) -> str:
        "Gets month from capture_at column"
        # Convert to datetime object and save month as int
        datetime_obj = datetime.datetime.fromtimestamp(captured_at/1000)
        month = int(datetime_obj.strftime("%m"))
        dates = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
        return dates[month]
    
    def _get_climate(self, climate_int: int) -> str:
        return climate_dict[climate_int]
    
    def _select_caption(self, index: int) -> str:
        """Generates a random caption for the given image using auxiliary data.

        EXAMPLE CAPTION:
            Location: A photo in the {CITY} city, {COUNTY} county, {STATE} state.
            Climate: This location has a temperate oceanic climate.
            Month: This photo was taken in December.

        Args:
            index (int): row index to generate caption for.

        Returns:
            str: randomly generated caption.
        """
        s = self.df.iloc[index]
        state = s.state
        city = s.city
        county = s['sub-region']
        # print(f'captured_at: {s.captured_at}, type: {type(s.captured_at)}')
        month = self._get_month(s.captured_at)
        climate = self._get_climate(int(s.climate))

        location_str = f"A photo in {city} city, {county}, in the state of {state}."
        climate_str = f"This location has a {climate} climate."
        month_str = f"This photo was taken in {month}."
        return location_str + ' ' + climate_str + ' ' + month_str
    
    def _crop_resize(self, image: Image.Image) -> Image.Image:
        """Crops and resizes the given image.
        
        Args:
            image (Image.Image): The image to be cropped and resized.
            
        Returns:
            Image.Image: The cropped and resized image.
        """
        # Crop the image to the largest possible square
        width, height = image.size
        new_dim = min(width, height)
        left = (width - new_dim) / 2
        top = (height - new_dim) / 2
        right = (width + new_dim) / 2
        bottom = (height + new_dim) / 2
        image = image.crop((left, top, right, bottom))

        # Resize the cropped image to a side length of self.image_size pixels
        return image.resize((self.image_size, self.image_size))

    def __getitem__(self, index: int) -> Tuple:
        """Retrieves item in dataset for given index.

        Args:
            index (int): sample index.

        Returns:
            Dict: sample model input
        """

        # Load the image
        id = self.df.iloc[index]['id']
        state = self.df.iloc[index]['state']
        image_filename = os.path.join(self.image_dir, state, str(id) + '.jpg')
        image = Image.open(image_filename)

        # Crop image
        image = self._crop_resize(image)

        # Generate a random caption for the image
        caption = self._select_caption(index)
        return image, caption
    
    def __len__(self):
        return len(self.df.index)

    def accuracy(self, model: Any, batch_size: int, trials: int=30) -> float:
        """Computes the accuracy of a given mode on the current dataset.

        Args:
            model (Any): pretrained CLIP model.
            batch_size (int): batch size of model
            trials (int, optional): Number of runs for the Monte-Carlo estimation
                of accuracy. Defaults to 30.

        Returns:
            float: accuracys
        """
        accs = []
        for t in range(trials):
            inputs = [self[(t * batch_size) + i] for i in range(batch_size)]
            images, captions = zip(*inputs)
            images = list(images)
            captions = list(captions)

            inputs = clip_processor(text=captions, images=images, return_tensors='pt',
                                    padding=True, truncation=True)
            for key in inputs:
                inputs[key] = inputs[key].to('cuda')

            inputs['return_loss'] = True
            outputs = model(**inputs)
            predictions = outputs.logits_per_image.softmax(dim=1).argmax(dim=1)
            accuracy = (predictions == torch.arange(batch_size, device='cuda')).sum()
            accs.append(accuracy / batch_size)
        
        acc = sum(accs) / trials
        return acc
    
# ds = PretrainDatasetOSVMini('train', 'datasets/osv-mini-129k')
# img, caption = ds[0]
# print(f'caption: {caption}')
# img.show()
# print(f'image size: {img.size}')

def read_unique_values(filepath: str) -> list:
    """Read unique values from a file, one value per line.
    
    Args:
        filepath (str): Path to file containing unique values
        
    Returns:
        list: Sorted list of unique values
    """
    with open(filepath, 'r') as f:
        values = [line.strip() for line in f.readlines()]
    return sorted(values)

class CLIPGeolocationDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, dir: str, unique_cities_file: str, unique_counties_file: str,
                 shuffle: bool=True, image_size: int=224):
        """Initialize the dataset.
        
        Args:
            split (str): Dataset split to load ('train', 'val', or 'test')
            dir (str): Path to parent directory containing the dataset
            unique_cities_file (str): Path to file containing unique city names
            unique_counties_file (str): Path to file containing unique county names
            shuffle (bool): Whether to shuffle the data
            image_size (int): Size to resize images to
        """
        self.split = split
        self.shuffle = shuffle
        self.image_size = image_size
        self.dir = dir
        self.image_dir = os.path.join(dir, f'{split}_images')
        self.csv_path = os.path.join(dir, f'{split}_mini.csv')

        # Basic checks
        assert split in ['train', 'val', 'test']
        assert os.path.exists(dir)
        assert os.path.exists(self.csv_path), f"CSV file does not exist: {self.csv_path}"
        assert os.path.exists(self.image_dir), f"Image directory does not exist: {self.image_dir}"
        assert os.path.exists(unique_cities_file), f"Cities file does not exist: {unique_cities_file}"
        assert os.path.exists(unique_counties_file), f"Counties file does not exist: {unique_counties_file}"
        
        # Load unique values
        self.unique_cities = read_unique_values(unique_cities_file)
        self.unique_counties = read_unique_values(unique_counties_file)
        
        # Load and preprocess data
        self.df = pd.read_csv(self.csv_path)
        self.df.drop(columns=["creator_username", "creator_id", 'thumb_original_url', 'sequence', 
                            "road_index", 'drive_side', 'soil'], inplace=True)
        self.df = self.df.rename(columns={'region': 'state'})
        
        # Convert month to 0-based index (1-12 -> 0-11) using epoch timestamp
        self.df['month'] = self.df['captured_at'].apply(
            lambda x: int(datetime.datetime.fromtimestamp(x/1000).strftime("%m")) - 1
        )
        
        # Ensure all required columns exist # TODO: remove this!!!
        required_columns = ['id', 'state', 'unique_sub-region', 'unique_city', 'climate', 'month', 
                          'latitude', 'longitude']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create label mappings using provided unique values
        self.state_to_idx = {state: idx for idx, state in enumerate(sorted(self.df['state'].unique()))}
        self.county_to_idx = {county: idx for idx, county in enumerate(self.unique_counties)}
        self.city_to_idx = {city: idx for idx, city in enumerate(self.unique_cities)}
        
        # Convert categorical labels to indices
        self.df['state_idx'] = self.df['state'].map(self.state_to_idx)
        self.df['county_idx'] = self.df['unique_sub-region'].map(self.county_to_idx)
        self.df['city_idx'] = self.df['unique_city'].map(self.city_to_idx)
        
        if shuffle:
            self.df = self.df.sample(frac=1.0, random_state=330)
            
        print(f'Initialized {split} dataset with {len(self.df)} samples')
        print(f'Number of unique states: {len(self.state_to_idx)}')
        print(f'Number of unique counties: {len(self.unique_counties)}')
        print(f'Number of unique cities: {len(self.unique_cities)}')
    
    def _crop_resize(self, image: Image.Image) -> Image.Image:
        """Crop and resize image to square."""
        width, height = image.size
        new_dim = min(width, height)
        left = (width - new_dim) / 2
        top = (height - new_dim) / 2
        right = (width + new_dim) / 2
        bottom = (height + new_dim) / 2
        image = image.crop((left, top, right, bottom))
        return image.resize((self.image_size, self.image_size))
    
    def _generate_caption(self, row: pd.Series) -> str:
        """Generate caption for CLIP training."""
        # Convert epoch timestamp to month name
        month = datetime.datetime.fromtimestamp(row['captured_at']/1000).strftime("%B")
        climate = climate_dict[int(row['climate'])]
        
        caption = (
            f"Location: A photo in {row['city']} city, {row['sub-region']} county, "
            f"{row['state']} state. Climate: This location has a {climate} climate. "
            f"Month: This photo was taken in {month}."
        )
        return caption
    
    def __getitem__(self, index: int) -> dict:
        """Get a single item from the dataset.
        
        Returns:
            dict: Dictionary containing:
                - pixel_values: Image tensor
                - input_ids: Text input IDs
                - attention_mask: Text attention mask
                - climate_labels: Climate classification label
                - month_labels: Month classification label
                - state_labels: State classification label
                - county_labels: County classification label
                - city_labels: City classification label
                - lat_labels: Latitude value
                - lng_labels: Longitude value
        """
        row = self.df.iloc[index]
        
        # Load and preprocess image
        image_path = os.path.join(self.image_dir, row['state'], f"{row['id']}.jpg")
        image = Image.open(image_path).convert('RGB')
        image = self._crop_resize(image)
        
        # Generate caption
        caption = self._generate_caption(row)
        
        # Process text with CLIP tokenizer
        text_inputs = clip_processor(
            text=caption,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        
        # Process image with CLIP processor
        image_inputs = clip_processor(
            images=image,
            return_tensors="pt"
        )

        # Prepare labels
        try:
            labels = {
                'climate_labels': torch.tensor(int(row['climate']), dtype=torch.long),
                'month_labels': torch.tensor(row['month'], dtype=torch.long),  # Already 0-based index
                'state_labels': torch.tensor(row['state_idx'], dtype=torch.long),
                'county_labels': torch.tensor(row['county_idx'], dtype=torch.long),
                'city_labels': torch.tensor(row['city_idx'], dtype=torch.long),
                'lat_labels': torch.tensor(float(row['latitude']), dtype=torch.float32),
                'lng_labels': torch.tensor(float(row['longitude']), dtype=torch.float32)
            }
        except Exception as e:
            print(f'\nERROR! {e}')
            print(f'row:\n {row}')
            raise e
        
        # Handle any potential NaN or invalid values
        for key, tensor in labels.items():
            if torch.isnan(tensor) or torch.isinf(tensor):
                # Replace NaN or inf values with a default value
                print(f'ERROR! {key} is NaN or inf')
                if 'lat' in key or 'lng' in key:
                    labels[key] = torch.tensor(0.0, dtype=torch.float32)
                else:
                    labels[key] = torch.tensor(0, dtype=torch.long)
        
        return {
            'pixel_values': image_inputs['pixel_values'].squeeze(0),
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            **labels
        }
    
    def __len__(self) -> int:
        """Get the total number of samples in the dataset."""
        return len(self.df)
