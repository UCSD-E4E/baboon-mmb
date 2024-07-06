function [Z, E] = InfaceExtFrankWolfe(X, gamma1, gamma2, MaxIter)
    delta = 10 * trace(X' * X); % Nuclear norm
    [m, n] = size(X);
    S = 1:m*n; % Index for known values
    xij = X(S); % Known values

    f = @(zij) 0.5 * sum((zij - xij).^2); % Objective function
    fGrad = zeros(m, n);
    Z = zeros(m, n);

    L = 1; % Lipschitz constant
    dm = 2 * delta;
    L_bar = L;
    dm_bar = dm;

    fGrad(S) = Z(S) - xij;
    [u, ~, v] = svds(fGrad, 1);

    Z = -delta * u * v';
    U = u; V = -v; D = delta;
    B = max(f(0) + trace(fGrad' * Z), 0);

    k = 0;
    err_est = 1;
    opt_met = 0.1;

    while err_est > opt_met && k < MaxIter
        fGrad(S) = Z(S) - xij;
        [u, ~, v] = svds(fGrad, 1);

        % In-face
        r = length(D);
        if r == min(m, n)
            Z_hat = delta * u * v';
            alpha_stop = binary_search(U, D, V, u, v, delta);
        else
            G = 0.5 * (V' * fGrad' * U + U' * fGrad * V);
            [ug, ~] = eigs(G, 1);
            ug = ug / norm(ug);
            Z_hat = U * delta * (ug * ug') * V';
            alpha_stop = (delta * ug' * D^-1 * ug - 1)^-1;
        end

        d = Z - Z_hat;
        Z_B = Z + alpha_stop * d;
        beta = min(-trace(fGrad' * d) / (L_bar * norm(d, 'fro')^2), alpha_stop);
        Z_A = Z + beta * d;

        if 1 / (f(Z_B(S)) - B) >= 1 / (f(Z(S)) - B) + gamma1 / (2 * L_bar * dm_bar^2)
            Z = Z_B;
            if r == min(m, n)
                [U, D, V] = svd_update(U, (1 + alpha_stop) * D, V, -alpha_stop * delta * u, v);
            else
                [R, D, R] = svd_thin((1 + alpha_stop) * D + alpha_stop * delta * (ug * ug'));
                U = U * R;
                V = V * R;
            end
        elseif 1 / (f(Z_A(S)) - B) >= 1 / (f(Z(S)) - B) + gamma2 / (2 * L_bar * dm_bar^2)
            Z = Z_A;
            if r == min(m, n)
                [U, D, V] = svd_update(U, (1 + beta) * D, V, -beta * delta * u, v);
            else
                [R, D, R] = svd_thin((1 + beta) * D + beta * delta * (ug * ug'));
                U = U * R;
                V = V * R;
            end
        else
            Z_tilda = -delta * u * v';
            Bw = f(Z(S)) + trace(fGrad' * (Z_tilda - Z));
            B = max(B, Bw);
            alpha = min(trace(fGrad' * (Z - Z_tilda)) / (L_bar * norm(Z - Z_tilda, 'fro')^2), 1);
            Z = Z + alpha * (Z_tilda - Z);
            [U, D, V] = svd_update(U, (1 - alpha) * D, V, -alpha * delta * u, v);
        end

        err_est = sum((X(:) - Z(:)).^2) / (m * n);
        k = k + 1;
    end

    E = X - Z;
end

function [u, d, v] = svd_thin(x)
    [u, d, v] = svd(x, 'econ');
    idx = find(diag(d) > 1e-6);
    u = u(:, idx);
    v = v(:, idx);
    d = d(idx, idx);
end

function [u, s, v] = svd_update(U, S, V, a, b)
    r = length(S);
    m = U' * a; p = a - U * m;
    p_norm = norm(p); P = p / p_norm;
    n = V' * b; q = b - V * n;
    q_norm = norm(q); Q = q / q_norm;
    K = [S, zeros(r, 1); zeros(1, r), 0] + [m; p_norm] * [n; q_norm]';
    [uk, s, vk] = svd_thin(K);
    u = [U, P] * uk;
    v = [V, Q] * vk;
end

function alpha_stop = binary_search(U, D, V, u, v, delta)
    alpha_stop = 0; alpha_max = 1;
    for i = 1:10
        alpha = (alpha_max - alpha_stop) / 2;
        [~, s, ~] = svd_update(U, (1 + alpha) * D, V, -alpha * delta * u, v);
        if sum(diag(s)) <= delta
            alpha_stop = alpha;
        else
            alpha_max = alpha;
        end
    end
end