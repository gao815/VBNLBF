function [s] = VBNLBF(B,L,M,FULL)
% VBNLBF - Vector Bayesian Beamformer with Noise Learning for MEG source imaging
% Implements the algorithm from:
%   "High-Res Brain Source Imaging of MEG using a Vector Bayesian Beamformer with Noise learning"
%
% Inputs:
%   B    : MEG sensor data [n_channels × n_timepoints]
%   L    : Leadfield matrix [n_channels × (n_dims * n_sources)] or [n_channels × n_dims × n_sources]
%   M    : Regularization matrix for source space (optional)
%   FULL : Flag for noise covariance structure (true = full, false = diagonal)
%
% Outputs:
%   s       : Estimated source time courses [n_sources*n_dims × n_timepoints]


%% ==================== Initialize parameters and variables ====================
% Get dimensions
nd = size(L,2)/size(M,1);      % Number of dimensions per source (typically 3)
iters = 400;                    % Maximum iterations
[nk, nt] = size(B);             % nk: number of sensors, nt: time points
nv = size(L,2)/nd;              % nv: number of sources

% Data scaling 
m = max(abs(B(:)));
B = B * (1/m) * 10;

% Reshape leadfield to 3D tensor [nk × nd × nv]
L = reshape(L, nk, nd, nv);
L = norm_leadf_col(L);          % Normalize leadfield columns

% Noise covariance configuration (Sec 2.1)
UPDATE_NOISE = true;
if FULL
    DIAG_NOISE = false;         % Full noise covariance (Eq. 4)
    FULL_NOISE = true;
else
    DIAG_NOISE = true;          % Diagonal noise covariance
    FULL_NOISE = false;
end

% Convergence thresholds (Sec 2.1)
MIN_DELTA_GAMMA = 1e-10;        % Gamma change threshold
MIN_GAMMA = 1e-8;               % Gamma pruning threshold
MIN_Z_EIGENVALUE = 1e-10;       % Minimum eigenvalue for Z_i

% Visualization flags
IF_PLOT = true;                 % Show convergence
SAVE_OUTPUT = false;            % Save output
Diag_Sub_Diag_Matrix_Update = 0;

%% Initialize variables
for i = 1:1:nd
    F(:,i,:) =  squeeze(L(:,i,:))/M;
end
F_2d = reshape(F,nk,nd*nv);
cost = zeros(iters+1);  
X = zeros(nd,nt,nv);    
s = zeros(nv*nd,nt);      
w = zeros(nd*nv,nk);
activeVoxelList = 1:nv;
inactiveVoxelList = [];
Gamma_init  = 1e-5*ones(nd,nd,nv)+rand(nd,nd,nv)*1e-3;
Gamma = Gamma_init;
Gamma_old = zeros(size(Gamma)); % Gamma of the previous iteration step
Sigma_n = eye(nk)*1e-1;

%% ==================== Main iteration loop ====================
for k=1:iters+1
    % Prune inactive sources
    tr = zeros(nv,1); 
    for i=1:nd 
        tr = tr + squeeze(Gamma(i,i,:));
    end
    if (min(tr) < MIN_GAMMA ) 
        % find active voxels
        activeVoxelList = find(tr > MIN_GAMMA)';
        inactiveVoxelList = setdiff(1:nv,activeVoxelList);
        % set Gamma of inactive voxels to zero
        for i=inactiveVoxelList
            Gamma(:,:,i)=zeros(nd,nd);
        end
        fprintf(' (pruned inactive voxels; %i of %i left (%i)\n',nv-size(activeVoxelList,2),nv,size(activeVoxelList,2));
    end
    
    %% calculate Sigma_b and its inverse
    SumLGammaLtrans = zeros(nk,nk);
    for i=activeVoxelList
        SumLGammaLtrans = SumLGammaLtrans+F(:,:,i)*Gamma(:,:,i)*F(:,:,i)';
    end
    Sigma_b = Sigma_n + SumLGammaLtrans; 
    invSigma_b = inv(Sigma_b);
    
    %% calculate cost function and deltaGamma
    cost(k) = sum(log(svd(Sigma_b))) + sum(sum( B.*(Sigma_b\B) ))/nt;
    dGamma = sum(sum(sum(abs(Gamma-Gamma_old))));
    
    %% plot cost function and voxel activity
    if IF_PLOT
        plotStatus(cost,iters,k,nv,nt,tr,activeVoxelList,X,Sigma_n)
        if SAVE_OUTPUT == 1; 
            export_fig('convergence','-pdf');
        end
    end
    
    %% exit conditions
    % for explanation, why this is in the middle of the loop, see comment 
    % after beginning of learning loop.
    if k>iters || dGamma < MIN_DELTA_GAMMA
        for i=inactiveVoxelList
            w(nd*i-nd+1:nd*i,:) = zeros(size(w(nd*i-nd+1:nd*i,:)));
        end
        break;
    end
    
    fprintf(' Loop %i/%i; deltaGamma = %f, cost = %f\n',k,iters,dGamma,cost(k));
    
    % sava Gamma of previous step
    Gamma_old=Gamma; 
    
    %% update sources
    for i=activeVoxelList
        % Update Diag Sub-Matrix method
        if ~Diag_Sub_Diag_Matrix_Update
            w(nd*i-nd+1:nd*i,:) = Gamma(:,:,i)*F(:,:,i)'*invSigma_b;
            X(:,:,i) = real(w(nd*i-nd+1:nd*i,:)*B); 
            Zi = F(:,:,i)'*invSigma_b*F(:,:,i); 
            [V, D] = eig(Zi); % eig decomp: Zi == V*D*V' for fast computation of inv and invsqrt, exploiting Zi beeing real and symmetric.
            sqrtZi = V*diag( sqrt((sign(diag(D))+1)/2.*diag(D)) )*V'; % sqrtm(Zi) == V*sqrtm(D)*V'
            D = diag(max(diag(D),MIN_Z_EIGENVALUE)); 
            invsqrtZi = V*diag( 1./(sqrt((sign(diag(D))+1)/2.*diag(D))) )*V'; % inv(sqrtm(Zi)) == V*inv(sqrtm(D))*V'
            Gamma(:,:,i) = real(invsqrtZi*sqrtmRealSym(sqrtZi*X(:,:,i)*X(:,:,i)'/nt*sqrtZi)*invsqrtZi);
        end
    end
    
    %% update sigma_n
    if UPDATE_NOISE
        if DIAG_NOISE == 0 && FULL_NOISE
            SumLGammaLtrans = zeros(nk,nk);
            for i=activeVoxelList
                SumLGammaLtrans = SumLGammaLtrans+F(:,:,i)*Gamma(:,:,i)*F(:,:,i)';
            end
            C = Sigma_n + SumLGammaLtrans; 
            Ls_est = F_2d*(w*B); 
            temp1 = B  - Ls_est;
            M = 1/nt*(temp1*temp1');
            eps_default = 1e-5; 
            [b_vec,b_val] = eig(C);
            root_C_coeff = sqrt(max(real(diag(b_val)),0));

            inv_root_C_coeff = zeros(size(C,1),1);
            inv_root_C_index = find(root_C_coeff >= eps_default);
            inv_root_C_coeff(inv_root_C_index) = 1./root_C_coeff(inv_root_C_index);

            root_C = b_vec * diag(root_C_coeff) * b_vec';
            inv_root_C = b_vec*diag(inv_root_C_coeff)*b_vec';

            [a_vec,a_val] = eig(root_C * M * root_C);
            A_coeff = sqrt(max(real(diag(a_val)),0));
            A = a_vec * diag(A_coeff) * a_vec';
            Sigma_n = (inv_root_C * A * inv_root_C);  
            if size(Sigma_n,1) ~= rank(Sigma_n)
                Sigma_n = Sigma_n + trace(Sigma_n)*0.01/size(Sigma_n,1)*eye(size(Sigma_n,1));
            end
        end
    end

    
end

%% Calculate final values and return

k = k-1; % subtract last step of the loop, because no update occurs there
cost = cost(1:k+1); % cut off placeholders in the cost vector for iterations
                    % never made
for i=1:nv % put all voxel time courses one below the other
    s((i*nd-nd+1):(i*nd),:) = X(:,:,i);   
end
% rescale sources and weights, because they where calculated not by y but
% by y/sqrt(nt).
w = w*sqrt(nt);
s = real(s)*sqrt(nt);
fprintf('\n\nPruned %i voxels.\n\n',(nv-size(activeVoxelList,2)));


%% Beamformer
LCMV_FLAG = 1;
if LCMV_FLAG
    lambda = 0;
    Sigma_b = Sigma_b + lambda*trace(Sigma_b)/size(Sigma_b,1)*eye(size(Sigma_b));
    inv_Sigma_b = inv(Sigma_b);
    for i= 1:nv
        W =  inv_Sigma_b*F(:,:,i)/(F(:,:,i)'*inv_Sigma_b*F(:,:,i));
        X(:,:,i) = real(W'*B);
        power(i) = trace(inv(F(:,:,i)'*inv_Sigma_b*F(:,:,i)));
    end
end
 s =  zeros(nv*nd,nt);
for i = 1:nv
    s((i-1)*nd+1:i*nd,:) = squeeze(X(:,:,i));    
end

return;
     
function A = sqrtmRealSym(A)
% This function performs an eigen decomposition and calculates the matrix
% square root through the scare root of the eigen values.
% If eigen values become negative (happens due to numerical reasons), they
% are set to zero.
[V D] = eig(A);  % eigen decomposition: A == V*D*V'
A = V*diag( sqrt((sign(diag(D))+1)/2.*diag(D)) )*V'; % A = sqrt(A)
% sign(diag(D))+1)/2 equals 1, if eigenvalue positive, 0 if negative and 1/2
% if 0, but in that case, the eigenvalue is zero anyway.
return;

                        
                        
function plotStatus(cost,iters,k,nv,nt,tr,activeVoxelList,X,Sigma_e)
    myColorOrder = sortrows(repmat([1 0 0;0 0 1; 1 1 0; 0 0 0],[3 1 1]));
    subplot(4,2,1), plot(0:k-1,real(cost(1:k)));
    title(['Cost function; ' int2str(k-1) ' / ' int2str(iters) ]);
    xlabel('Number of iterations');
    ylabel('Cost');
    set(gca(),'XLim',[0 k]);

    subplot(4,2,2), plot(1:nv,tr);
    title(['trace(Gamma); active voxels: ' int2str(size(activeVoxelList,2)) ' / ' int2str(nv)]);
    xlabel('Voxels i');
    ylabel('trace(Gamma_i)');
    if size(X,1) > 1
        tr = sum(abs(squeeze(mean(X.^2,1))),1)';
    end
    % find and plot most active voxels
    mostActiveVoxels = find(tr>0.1*max(tr));
    [neverused,index] = sort(tr(mostActiveVoxels),'descend');
    for i=1:min(5,size(mostActiveVoxels,1))
        v = mostActiveVoxels(index(i));
        subplot(4,2,3:4);%, plot(1:nt,X(:,:,v)');
        X_power = squeeze(mean(X.^2,1));
        plot(1:nt,X_power(:,v))
        hold on;
    end
    hold off;
    
    title('source time courses)');
    xlabel('time (samples)');
    ylabel('source strength');
    subplot(4,2,5:6);
    bar(sum(abs(squeeze(mean(X.^2,1))),1)');
    drawnow;
    %maximize; % maximizes the figure window after plotting. requires the 
              % maximize function from matlabcentral. no effect on the results.
              
    subplot(4,2,7);
    imagesc(real(Sigma_e));
    drawnow;
    
    return;                       
    
function [L,colnorm] = norm_leadf_col(L)

    L = permute(L,[1 3 2]);

    if size(L,3)>1
        for i=1:size(L,3)
            colnorm(:,i) = sqrt(sum(L(:,:,i).^2));
            L(:,:,i) = L(:,:,i)./repmat(colnorm(:,i)',[size(L,1) 1]);
        end
    else
        for i=1:size(L,2)
            colnorm(:,i) = sqrt(sum(L(:,i).^2));
            L(:,i) = L(:,i)./repmat(colnorm(:,i)',[size(L,1) 1]);
        end
    end

    L = permute(L,[1 3 2]);
