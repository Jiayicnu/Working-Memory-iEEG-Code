%% SVM classification after residual-based regression for EEG data.
% This function removes independent EEG activity from dependent EEG signals 
% and uses SVM to classify the remaining activity.
% Input:
%   - EEG_Indep: struct, independent EEG data of load6 and load8 (trials x freq x time)
%   - EEG_Dep: struct, dependent EEG data of load6 and load8 
% Output:
%   - accuracy: vector (100 x 1), classification accuracy across 100 iterations.
% Requires LIBSVM for SVM training and testing.

function accuracy = SVM_residual(EEG_Indep, EEG_Dep)
   
   % Compute residuals for Load6
   for i=1:size(EEG_Indep.load6,1)
        x = EEG_Indep.load6(i,:,:); 
        y = EEG_Dep.load6(i,:,:);
        x = x(:); 
        y = y(:);
        X = [ones(length(y),1), x];
        [~, residuals] = regress(y, X); 
        res_load6(i, :) = residuals';
   end
   
   % Compute residuals for Load8
   for i=1:size(EEG_Indep.load8,1)
        x = EEG_Indep.load8(i,:,:); 
        y = EEG_Dep.load8(i,:,:);
        x = x(:); 
        y = y(:);
        X = [ones(length(y),1), x]; 
        [~, residuals] = regress(y, X); 
        res_load8(i, :) = residuals';
   end 
   
    % define number of cross-validation
    num = 100;
    % SVM training and testing
    for i = 1:num
        % Split 70% training, 30% testing
        [train_idx6, ~, test_idx6] = dividerand(size(res_load6,1), 0.7, 0, 0.3);
        [train_idx8, ~, test_idx8] = dividerand(size(res_load8,1), 0.7, 0, 0.3);
        train_set6 = res_load6(train_idx6, :);
        test_set6 = res_load6(test_idx6, :);
        train_set8 = res_load8(train_idx8, :);
        test_set8 = res_load8(test_idx8, :);
        train = [train_set6; train_set8];
        test = [test_set6; test_set8];
        train_label = [ones(size(train_set6,1),1); -ones(size(train_set8,1),1)];
        test_label = [ones(size(test_set6,1),1); -ones(size(test_set8,1),1)];

        % Normalize data
        [Train, PS] = mapminmax(train');
        train_zscore = Train';
        Test = mapminmax('apply', test', PS);
        test_zscore = Test';
        [pc, score, latent, tsquare] = pca(train_zscore, 'Algorithm', 'svd', 'Centered', false);
        tmp = cumsum(latent) ./ sum(latent);
        indd = find(tmp >= 0.99);
        train_pc = train_zscore * pc(:, 1:indd(1));
        test_pc = test_zscore * pc(:, 1:indd(1));

        % SVM training & prediction
        parameter = ['-s ' num2str(s) ' -t ' num2str(t) ' -c ' num2str(1) ' -g ' num2str(0.1) ' -b 1'];        
        model = svmtrain(train_label, train_pc, parameter);
        [~, acc, ~] = svmpredict(test_label, test_pc, model);
        accuracy(i) = acc(1);
   end
end