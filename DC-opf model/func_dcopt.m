function [blackoutSize, flag] = func_dcopt(systemState, mpc0)
    %{
      func_dcopt.m outputs the relative blackout size (in percentage),
      given current network state, network after load dispatch (mpc) and
      network before load dispatch (mpc0)

      Compatible with MATPOWER v7.1+ and Octave.
    %}
    %% ensure plain struct format (MATPOWER v8+ uses mp_table objects)
    if isstruct(mpc0) && isfield(mpc0, 'version')
        mpc0 = struct(mpc0);
        if ~isnumeric(mpc0.bus)
            mpc0.bus     = double(mpc0.bus);
            mpc0.gen     = double(mpc0.gen);
            mpc0.branch  = double(mpc0.branch);
            if isfield(mpc0, 'gencost')
                mpc0.gencost = double(mpc0.gencost);
            end
        end
    end

    %% basic info. of the network
    busDic     = mpc0.bus(:, 1);      % bus no.
    genDic     = mpc0.gen(:, 1);      % generator no.
    branchDic  = mpc0.branch(:, 1:2); % end bus no. of a each branch

    nb = size(busDic, 1);       % #buses
    ng = size(genDic, 1);       % #generators
    nl = size(branchDic, 1);    % #branches

    %% network configuration
    busState    = reshape(systemState(1:nb), [nb, 1]);
    branchState = reshape(systemState(nb+1:nb+nl), [nl, 1]);

    [~, idx_g] = ismember(genDic, busDic); % idx_g: index of the generator in busDic
    genState   = busState(idx_g);

    %% update the matpower model considering component failure

    % convert the fixed load to dispatchable load
    % note that the gencost, gen, and bus table have been changed in this process
    mpc = load2disp(mpc0);

    % ensure plain numeric matrices after load2disp (MATPOWER v8+)
    mpc.bus     = double(mpc.bus);
    mpc.gen     = double(mpc.gen);
    mpc.branch  = double(mpc.branch);
    mpc.gencost = double(mpc.gencost);

    mpc.gencost(1:ng, :) = repmat([2, 0, 0, 2, 0, 0, 0], ng, 1); % no cost for the generator

    % for better convergence of DC-opf, we delete the shunt conductance and
    % undispatchable load for each bus, and set the minimal real power
    % generation for each bus to 0
    mpc.bus(:, 3)     = 0;
    mpc.bus(:, 5)     = 0;
    mpc.gen(1:ng, 10) = 0;
    mpc.gen(1:ng, 8)  = 1;               % let all generators be in service
    totpf = - sum(mpc.gen(ng+1:end, 2)); % total real power demand

    % update bus table
    removedBusIdx = find( busState == 1 );
    removedBusNo  = busDic( removedBusIdx );

    mpc.bus(removedBusIdx, :) = [];

    % update generator and gencost table
    mpc.gen(1:ng, 4:5)  = (1-genState).*mpc.gen(1:ng, 4:5);
    mpc.gen(1:ng, 9:10) = (1-genState).*mpc.gen(1:ng, 9:10);

    t = ismember(mpc.gen(:, 1), removedBusNo);
    mpc.gen(t, :)     = [];
    mpc.gencost(t, :) = [];

    % update branch table
    removedBranchIdx = find( branchState == 1 );
    removedBranchIdx = union( removedBranchIdx, find(ismember( branchDic(:, 1), removedBusNo )==1) );
    removedBranchIdx = union( removedBranchIdx, find(ismember( branchDic(:, 2), removedBusNo )==1) );
    removedBranch    = branchDic(removedBranchIdx, :);

    mpc.branch(removedBranchIdx, :) = [];

    % setting the options
    mpopt  = mpoption('out.all', 0, 'verbose', 0);
    mpopt.exp.use_legacy_core = 1;  % bypass MP-Core (required for MATPOWER v8+)

    % run the solver
    results = rundcopf(mpc, mpopt);

    if results.success ~= 1
        LogFile = 'log.txt';
        fID = fopen(LogFile, 'a');
        for i = 1:nb+nl
            fprintf(fID, [num2str(systemState(i) ), ' ']);
        end
        fprintf(fID, '#\n');
        fclose(fID);
    end

    flag   = results.success;

    realpf = (mpc.gen( 1:sum(genState ~= 1), 8))' * results.gen(1:sum(genState ~= 1), 2);
    blackoutSize = 100 * round( abs(realpf - totpf)/totpf * 1e8 ) / 1e8;

end