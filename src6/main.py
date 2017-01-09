import data_processing
import nn_model as tmm
import numpy as np
import database as rid
import sys

# Run with: python main [-split=0.2] [-epoch=15]
nb_epoch, valid_split, deep_id_dim, aux_weight = 1, 0.9, 2000, 0.1
model_name, query_res = 'my_model1', 'try.xml'


# parsing CMD arguments
for arg in sys.argv:
    if arg.startswith('-split='):
        valid_split = float(arg[len('-split='):])
        if valid_split > 1.0:
            valid_split = int(valid_split)
    elif arg.startswith('-epoch='):
        nb_epoch = int(arg[len('-epoch='):])
    elif arg.startswith('-deepID-dim='):
        deep_id_dim = int(arg[len('-deepID-dim='):])
    elif arg.startswith('-aux-weight='):
        aux_weight = float(arg[len('-aux-weight='):])
    elif arg.startswith('-model-name='):
        model_name = arg[len('-model-name='):]
    elif arg.startswith('-query-res='):
        query_res = arg[len('-query-res='):]


print('Hyper parameters configuration:')
print(nb_epoch, valid_split, deep_id_dim, aux_weight, model_name, query_res)

PREPROCESSED = True
LOAD_TRAINED_MODEL = False
if __name__ == '__main__':
    dp = data_processing.DataProcesser()
    if not PREPROCESSED:
        dp.equalize_hist_all('../data/val/')
        dp.equalize_hist_all('../data/train/')

    # Note that y_dev and y_val are used to index cars with same VID, not a target to be predicted
    (X_dev, y_dev), (X_val, y_val), nb_classes = dp.load_data(validation_split=valid_split)
    # TODO: WHAT IF CHANGE y_dev TO SOMETHING TO BE PREDICTED?
    # SAY, DIFFERENCE BETWEEN TWO PAIRS? (METRIC LEARNING)
    (X_dev1, X_dev2, X_dev3), (y_dev, X_dev1_vid) = dp.make_triplet(X_dev, y_dev, nb_classes)

    model = tmm.TripletModel(deep_id_dim, aux_weight, nb_epoch, nb_classes, model_name)
    if LOAD_TRAINED_MODEL:
        model.load_model()
    else:
        # Add target later
        model.fit([X_dev1, X_dev2, X_dev3], [y_dev, X_dev1_vid])
        model.save_model()

    # Use validation set to measure generality of our model
    deep_id_val_ref, deep_id_val_query = model.get_deep_id(X_dev1), model.get_deep_id(X_val)
    print deep_id_val_ref.shape, deep_id_val_query.shape
    np.save('../data/res/deep_id_val_ref.npy', deep_id_val_ref)
    np.save('../data/res/deep_id_val_query.npy', deep_id_val_query)
    db1 = rid.MyDataBase(deep_id_val_ref)
    result = db1.retrieve_on_batch(deep_id_val_query)
    MAP, AP = db1.calc_MAP(result, [map(lambda t: t[1], same_car_list) for same_car_list in y_val])
    print '****** MAP =', MAP, '*******'
    print 'AP =', AP
    np.save('../data/res/res_val.npy', result)
    print result

    # Run model on test set to generate re-identification result to submit to the system
    X_ref, X_query, query_image_names = dp.load_test_data()
    deep_id_ref, deep_id_query = model.get_deep_id(X_ref), model.get_deep_id(X_query)
    print deep_id_ref.shape, deep_id_query.shape
    np.save('../data/res/deep_id_test_ref.npy', deep_id_ref)
    np.save('../data/res/deep_id_test_query.npy', deep_id_query)
    db2 = rid.MyDataBase(deep_id_ref)
    result = db2.retrieve_on_batch(deep_id_query)
    result_name_list = dp.indices2image_name(result)
    db2.printXML(query_image_names, result_name_list, '../data/res/' + query_res)
    np.save('../data/res/res_test.npy', result)
    print result

print('Hyper parameters summary (again):')
print(nb_epoch, valid_split, deep_id_dim, aux_weight, model_name, query_res)
