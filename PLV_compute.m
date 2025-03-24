%% Compute phase locking value (PLV)
% Inputs:
%   - data: EEG data (channels * time * trials)
%   - Fre: Frequency range (e.g., 1:100)
%   - Fsample: Sampling rate (e.g., 1000 Hz)
%   - wavenum: Wavelet transform parameter (e.g., 6)
% Output:
%   - PLV: Computed PLV (channels * channels * frequencies * time)
% subfunctions: BOSC_tf (function for time-frequency decomposition)
function PLV = PLV_compute(data, Fre, Fsample, wavenum)

    [nchan, npts, ntrials] = size(data);
    spectrum = zeros(ntrials, nchan, length(Fre), npts);

    for i = 1:nchan
        for j = 1:ntrials
            tmp = data(i, :, j);
            [B, ~, ~] = BOSC_tf(tmp, Fre, Fsample, wavenum);
            spectrum(j, i, :, :) = B;
        end
    end

    PLV = zeros(nchan, nchan, length(Fre), size(data, 2));
    for e1 = 1:nchan-1
        for e2 = e1+1:nchan
            sig1 = angle(squeeze(spectrum(:, e1, :, :))); 
            sig2 = angle(squeeze(spectrum(:, e2, :, :)));
            e = exp(1i * (sig1 - sig2));
            plv = abs(sum(e, 1)) / ntrials;
            PLV(e1, e2, :, :) = plv;
            PLV(e2, e1, :, :) = plv;
        end
    end
end