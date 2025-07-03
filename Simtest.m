% Call VBNLBF algorithm for MEG source imaging
s = VBNLBF(B, lf, sparse(eye(Nsource_scale)), 1); 
% Parameters:
%   B: MEG sensor data [n_channels × n_timepoints]
%   lf: Leadfield matrix [n_channels × n_dims × n_sources] 
%   sparse(eye(Nsource_scale)): Source space regularization matrix (identity form)
%   1: FULL flag indicating full noise covariance structure
%
% Output:
%   s: Estimated source time courses [n_sources*n_dims × n_timepoints]

% Take real part (algorithm may produce complex values, take real part for practical use)
s = real(s);

% Initialize source power matrix [n_sources × n_timepoints]
S = zeros(Nsource_scale, size(s,2));

% Loop through all sources
for i = 1:Nsource_scale
    % Check source dimensionality
    if Nsource_dim > 1
        % Calculate vector magnitude for current source (Eq.7 in paper)
        % Extract current source time courses [n_dims × n_timepoints]
        start_idx = (i-1)*Nsource_dim + 1;
        end_idx = i*Nsource_dim;
        source_vector = s(start_idx:end_idx, :);
        
        % Compute vector magnitude: sqrt(s_x^2 + s_y^2 + s_z^2)
        S(i, :) = sqrt(sum(source_vector.^2, 1));
    end
end

% Final outputs:
%   s: Vector source time courses [n_sources*n_dims × n_timepoints]
%        (contains directional information)
%   S: Source power time courses [n_sources × n_timepoints]
%        (magnitude of neural activity at each source)