% CROSS_REGION_DECODING: Train a classifier on one brain region and test on another
% Requires LIBSVM package proposed by Chang et al.,2011.
% The procedure is very similar to decode load 6 vs load 8.
% Input:
%   data_train (struct): Training data structure with fields:
%       - data_set4: Training data for load 4 (num_trials x num_freqs x num_times).
%       - data_set6: Training data for load 6 
%   data_test (struct): Testing data structure with fields:
%       - data_set4: Test data for load 4 
%       - data_set6: Test data for load 6
% Output:
%   accuracy (vector): Classification accuracies for each cross-validation iteration (length = 100).
% Example:
%   accuracy = SVM_crossregion(data_train, data_test);
function accuracy = SVM_crossregion(data_train,data_test)  

    % trainarea data
    trainarea_set4 = data_train.data_set4; 
    trainarea_set6 = data_train.data_set6; 
    trainarea_set4 = reshape(trainarea_set4, [size(trainarea_set4,1) size(trainarea_set4,2)*size(trainarea_set4,3)]);
    trainarea_set6 = reshape(trainarea_set6, [size(trainarea_set6,1) size(trainarea_set6,2)*size(trainarea_set6,3)]);

    % testarea data
    testarea_set4 = data_test.data_set4; 
    testarea_set6 = data_test.data_set6; 
    testarea_set4 = reshape(testarea_set4, [size(testarea_set4,1) size(testarea_set4,2)*size(testarea_set4,3)]);
    testarea_set6 = reshape(testarea_set6, [size(testarea_set6,1) size(testarea_set6,2)*size(testarea_set6,3)]);

    % define number of cross-validation
    num = 100;

    % SVM training and testing
    for i = 1:num
        % Randomly split 70% for training and 30% for testing
        [train_set4_idx(i,:),~,test_set4_idx(i,:)] = dividerand(size(trainarea_set4,1), 0.7, 0, 0.3);
        test_set4 = testarea_set4(test_set4_idx(i,:), :);
        train_set4 = trainarea_set4(train_set4_idx(i,:), :);
        [train_set6_idx(i,:),~,test_set6_idx(i,:)] = dividerand(size(trainarea_set6,1), 0.7, 0, 0.3);
        test_set6 = testarea_set6(test_set6_idx(i,:), :);
        train_set6 = trainarea_set6(train_set6_idx(i,:), :);
        test = [test_set4; test_set6];
        test_label = [1*ones(size(test_set4,1),1); -1*ones(size(test_set6,1),1)];
        train_set4_label = 1*ones(size(train_set4,1),1);
        train_set6_label = -1*ones(size(train_set6,1),1);
        train = [train_set4; train_set6];
        train_label = [train_set4_label; train_set6_label];

        % Normalize the training and testing data
        [Train, PS] = mapminmax(train');
        train_zscore = Train';
        Test = mapminmax('apply', test', PS);
        test_zscore = Test';

        % Perform PCA to reduce dimensionality
        [pc, score, latent, tsquare] = pca(train_zscore, 'Algorithm', 'svd', 'Centered', false);
        tmp = cumsum(latent) ./ sum(latent);
        indd = find(tmp >= 0.99);
        train_pc = train_zscore * pc(:, 1:indd(1));
        test_pc = test_zscore * pc(:, 1:indd(1));

        % SVM classifier
        parameter = ['-s ' num2str(s) ' -t ' num2str(t) ' -c ' num2str(1) ' -g ' num2str(0.1) ' -b 1'];
        model = svmtrain(train_label, train_pc, parameter);
        [predict_label, acc, dec_values] = svmpredict(test_label, test_pc, model);
        accuracy(i,1) = acc(1);
    end

end