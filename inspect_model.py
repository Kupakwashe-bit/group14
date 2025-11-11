import h5py
import numpy as np

def inspect_model_architecture(model_path):
    print(f"Inspecting model: {model_path}")
    
    try:
        # Open the HDF5 file
        with h5py.File(model_path, 'r') as f:
            print("\nModel keys:", list(f.keys()))
            
            # Check for model configuration
            if 'model_weights' in f:
                print("\nModel weights structure:")
                def print_weights(name, obj):
                    if isinstance(obj, h5py.Group):
                        print(f"  Group: {name}")
                        # Print the first level of attributes
                        if obj.attrs:
                            print("    Attributes:")
                            for k, v in obj.attrs.items():
                                print(f"      {k}: {v}")
                    elif isinstance(obj, h5py.Dataset):
                        print(f"  Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
                
                f['model_weights'].visititems(print_weights)
            
            # Check for model configuration
            if 'model_config' in f:
                print("\nModel configuration:")
                print(f.attrs['model_config'])
            
            # Check for training configuration
            if 'training_config' in f:
                print("\nTraining configuration:")
                print(f.attrs['training_config'])
                
    except Exception as e:
        print(f"Error inspecting model: {e}")

if __name__ == "__main__":
    model_path = 'models/cifar10_model.h5'
    inspect_model_architecture(model_path)
