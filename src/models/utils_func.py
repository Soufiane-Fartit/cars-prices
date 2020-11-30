import string
import random

def id_generator(size=6, chars=string.ascii_lowercase + string.digits):
    """GENERATE A RANDOM STRING TO BE USED AS AN ID

    Args:
        size (int, optional): size of the string. Defaults to 6.
        chars (str, optional): charachters to be used to generate the string. 
                                Defaults to string.ascii_lowercase+string.digits.

    Returns:
        [str]: a random chain of charachters
    """
    return ''.join(random.choice(chars) for _ in range(size))



def save_model(path, model):
    """SAVE MODEL INTO PICKLE FILE

    Args:
        path (str): path where to save the model
        model (binary): the model to be saved
    """
    import pickle

    with open(path, 'wb') as file:
        pickle.dump(model, file)



def update_history(models_hist_path, model_id, model_name, model, params):
        """SAVE METADATA RELATED TO THE TRAINED MODEL INTO THE HISTORY FILE

        Args:
            models_hist_path (str): path to the history file
            model_id (str): unique id of the model
            model_name (str): model name = "model_"+model_id+".pkl"
            model (binary): binary file of the model
            params (dict): dictionnary containing the hyper-parameters 
                            used to fit the model
        """
        from datetime import datetime
        import json

        model_metadata = dict()
        model_metadata['trained'] = str(datetime.now())
        model_metadata['model_type'] = type(model).__name__
        model_metadata['model_id'] = model_id
        model_metadata['params'] = params
        print(model_metadata)

        with open(models_hist_path, 'r+') as f:
            try :
                hist = json.load(f)
                hist[model_name] = model_metadata
                f.seek(0)
                json.dump(hist, f, indent=4)
            except json.decoder.JSONDecodeError:
                json.dump({model_name:model_metadata}, f, indent=4)



def update_history_add_eval(models_hist_path, model_id=None, model_name=None, metrics=None):
    """ADD EVALUATION METRICS THE HISTORY FILE FOR THE SPECIFIED MODEL

    Args:
        models_hist_path (str): path to the history file
        model_id (str, optional): the id of the model. Defaults to None.
        model_name (str, optional): the name of the model. Defaults to None.
        metrics (dict, optional): a dictionnary containing metadata related 
                                    to the model evaluation. Defaults to None.
    """
        from datetime import datetime
        import json
        
        assert model_id!=None or model_name!=None, "At least the model id or name must be given"
        assert models_hist_path!=None, "You must specify the path to the history file"

        if not model_name:
            model_name = "model_"+model_id+".pkl"

        eval_metadata = dict()
        eval_metadata['datetime'] = str(datetime.now())
        eval_metadata['metrics'] = metrics

        with open(models_hist_path, 'r+') as f:
            try :
                hist = json.load(f)
                hist[model_name]['evaluation'] = eval_metadata
                f.seek(0)
                json.dump(hist, f, indent=4)
            except json.decoder.JSONDecodeError:
                print('cannot save evaluation metadata')
