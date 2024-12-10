import numpy as np
from scipy.optimize import OptimizeResult

def _minimize_scalar_bounded(func, bounds, global_max_fun, get_global_fun_calls, args=(),
                             xatol=1e-5, maxiter=500, disp=0,
                             **unknown_options):
    """
    Minimiza a função `func` dentro do intervalo `bounds` usando busca da seção áurea
    e interpolação parabólica.

    Parameters
    ----------
    func : callable
        Função a ser minimizada.
    bounds : tuple
        Intervalo (x1, x2) para a busca.
    args : tuple, optional
        Argumentos adicionais para a função `func`.
    xatol : float, optional
        Tolerância absoluta para a solução.
    maxiter : int, optional
        Número máximo de iterações.
    disp : int, optional
        Controle de impressão de mensagens de status.
    """
    _check_unknown_options(unknown_options)
    maxfun = maxiter

    # Verifica se os limites do intervalo são válidos
    if len(bounds) != 2:
        raise ValueError('bounds must have two elements.')
    x1, x2 = bounds
    if not (is_finite_scalar(x1) and is_finite_scalar(x2)):
        raise ValueError("Optimization bounds must be finite scalars.")
    if x1 > x2:
        raise ValueError("The lower bound exceeds the upper bound.")

    # Inicializa variáveis e constantes
    flag = 0
    sqrt_eps = np.sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - np.sqrt(5.0))
    a, b = x1, x2
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = 0.0
    x = xf
    fx = func(x, *args)
    num = 1
    fmin_data = (1, xf, fx)
    fu = np.inf
    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1

    if disp > 2:
        print(" ")
        print(' Func-count     x          f(x)          Procedure')
        print("%5.0f   %12.6g %12.6g %s" % (fmin_data + ('initial',)))

    while (np.abs(xf - xm) > (tol2 - 0.5 * (b - a))):
        golden = 1
        # Verifica ajuste parabólico
        # if np.abs(e) > tol1:
        #     golden = 0
        #     r = (xf - nfc) * (fx - ffulc)
        #     q = (xf - fulc) * (fx - fnfc)
        #     p = (xf - fulc) * q - (xf - nfc) * r
        #     q = 2.0 * (q - r)
        #     if q > 0.0:
        #         p = -p
        #     q = np.abs(q)
        #     r = e
        #     e = rat

        #     # Verifica aceitabilidade da parábola
        #     if ((np.abs(p) < np.abs(0.5*q*r)) and (p > q*(a - xf)) and
        #             (p < q * (b - xf))):
        #         rat = (p + 0.0) / q
        #         x = xf + rat
        #         step = '       parabolic'
        #         if ((x - a) < tol2) or ((b - x) < tol2):
        #             si = np.sign(xm - xf) + ((xm - xf) == 0)
        #             rat = tol1 * si
        #     else:
        #         golden = 1

        if golden:
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean * e
            step = '       golden'

        si = np.sign(rat) + (rat == 0)
        x = xf + si * np.maximum(np.abs(rat), tol1)
        fu = func(x, *args)
        num += 1
        fmin_data = (num, x, fu)
        if disp > 2:
            print("%5.0f   %12.6g %12.6g %s" % (fmin_data + (step,)))

        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        # global_fun_calls = get_global_fun_calls()
        # print(f"global_fun_calls: {global_fun_calls} global_max_fun: {global_max_fun}")
        # if num >= maxfun or global_fun_calls >= global_max_fun:
        if num >= maxfun:
            # print(f"parei o line com global_fun_calls: {global_fun_calls}")
            flag = 1
            break

    if np.isnan(xf) or np.isnan(fx) or np.isnan(fu):
        flag = 2

    fval = fx
    if disp > 0:
        _endprint(x, flag, fval, maxfun, xatol, disp)

    result = OptimizeResult(fun=fval, status=flag, success=(flag == 0),
                            message={0: 'Solution found.',
                                     1: 'Maximum number of function calls reached.',
                                     2: 'NaN encountered.'}.get(flag, ''),
                            x=xf, nfev=num, nit=num)

    return result

def _check_unknown_options(unknown_options):
    if unknown_options:
        raise ValueError("Unknown options: %s" % unknown_options)

def is_finite_scalar(x):
    return np.isfinite(x) and np.isscalar(x)

def _endprint(x, flag, fval, maxfun, xatol, disp):
    if flag == 0:
        msg = "Solution found."
    elif flag == 1:
        msg = "Maximum number of function calls reached."
    elif flag == 2:
        msg = "NaN encountered."
    else:
        msg = "Unknown error."
    print("Optimization terminated successfully.")
    print("         Current function value: %f" % fval)
    print("         Iterations: %d" % maxfun)
    print("         Function evaluations: %d" % maxfun)

def _line_for_search(x0, alpha, lower_bound, upper_bound):
    """
    Given a parameter vector ``x0`` with length ``n`` and a direction
    vector ``alpha`` with length ``n``, and lower and upper bounds on
    each of the ``n`` parameters, what are the bounds on a scalar
    ``l`` such that ``lower_bound <= x0 + alpha * l <= upper_bound``.


    Parameters
    ----------
    x0 : np.array.
        The vector representing the current location.
        Note ``np.shape(x0) == (n,)``.
    alpha : np.array.
        The vector representing the direction.
        Note ``np.shape(alpha) == (n,)``.
    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.

    Returns
    -------
    res : tuple ``(lmin, lmax)``
        The bounds for ``l`` such that
            ``lower_bound[i] <= x0[i] + alpha[i] * l <= upper_bound[i]``
        for all ``i``.

    """
    # get nonzero indices of alpha so we don't get any zero division errors.
    # alpha will not be all zero, since it is called from _linesearch_powell
    # where we have a check for this.
    nonzero, = alpha.nonzero()
    lower_bound, upper_bound = lower_bound[nonzero], upper_bound[nonzero]
    x0, alpha = x0[nonzero], alpha[nonzero]
    low = (lower_bound - x0) / alpha
    high = (upper_bound - x0) / alpha

    # positive and negative indices
    pos = alpha > 0

    lmin_pos = np.where(pos, low, 0)
    lmin_neg = np.where(pos, 0, high)
    lmax_pos = np.where(pos, high, 0)
    lmax_neg = np.where(pos, 0, low)

    lmin = np.max(lmin_pos + lmin_neg)
    lmax = np.min(lmax_pos + lmax_neg)

    # if x0 is outside the bounds, then it is possible that there is
    # no way to get back in the bounds for the parameters being updated
    # with the current direction alpha.
    # when this happens, lmax < lmin.
    # If this is the case, then we can just return (0, 0)
    return (lmin, lmax) if lmax >= lmin else (0, 0)