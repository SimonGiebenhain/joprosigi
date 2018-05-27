def predict():
    model.load_weights('weights/every_epoch.hdf5')
    batch_size_test = 50
    zip_path_test = '../data/test_jpg.zip'
    test_df = pd.read_csv('../data/nn_test.csv')
    all_test_df = np.array_split(test_df,10)
    pred_list = []
    for i in range(10):
        seq_1 = data_sequence_test(all_test_df[i], batch_size=batch_size_test, zip_path=zip_path_test)
        preds = model.predict_generator(seq_1, workers=4,use_multiprocessing=True, verbose=1)
        np.save('../data/nn_predictions_'+ str(i) + '.npy' , preds)
        pred_list.append(preds)
    submission = pd.read_csv('../data/nn_sample_submission.csv')
    submission['deal_probability'] = np.concatenate(pred_list, axis=0)
    submission.to_csv("submission.csv", index=False)