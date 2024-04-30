from jax.numpy import arange, exp, log, ndim, pi, sqrt, sin, abs, where, array

a = 12.5
ks = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]  # arange(1, a)

c_0 = 2.50662827463100024161235523934010416269302368164062
c_k = array(
    [
        334761.46676680451491847634315490722656250000000000000000,
        -1235594.61986167938448488712310791015625000000000000000000,
        1858132.14120021951384842395782470703125000000000000000000,
        -1466621.63011462800204753875732421875000000000000000000000,
        652789.34603856515605002641677856445312500000000000000000,
        -163966.96116112949675880372524261474609375000000000000000,
        22061.98757932552689453586935997009277343750000000000000,
        -1415.75624313534285647619981318712234497070312500000000,
        34.60099121429649926540150772780179977416992187500000,
        -0.20248984794909297146858762062038294970989227294922,
        0.00008722418296303442947561834763803290115902200341,
        -0.00000000001426088812090673789714138169595669712944,
    ]
)


def gamma_spouge(z):
    sum = c_0
    count = 0
    for k in ks:
        sum += c_k[count] / (z - 1 + k)
        count += 1
    return (z - 1 + a) ** (z - 0.5) * exp(-z + 1 - a) * sum


def loggamma(z):
    return log(gamma_spouge(z))


"""
from numpy import arange, exp, log, ndim, pi, sqrt
from scipy.special import gamma
try:
    from scipy.special import loggamma
except ImportError:
    def loggamma(x):
        return log(gamma(x))
"""


def _deriv(MK, deriv):
    """Real deriv is wrt :math:`t`, complex deriv is wrt :math:`\ln t`"""
    if deriv == 0:
        return MK

    if isinstance(deriv, complex):

        def MKderiv(z):
            return (-z) ** deriv.imag * MK(z)

        return MKderiv

    def MKderiv(z):
        poly = arange(deriv) + 1
        poly = poly - z if ndim(z) == 0 else poly - z.reshape(-1, 1)
        poly = poly.prod(axis=-1)
        return poly * MK(z - deriv)

    return MKderiv


def Mellin_BesselJ(nu, deriv=0):
    def MK(z):
        """
        return exp(
            log(2) * (z - 1) + loggamma(0.5 * (nu + z)) - loggamma(0.5 * (2 + nu - z))
        )
        """
        return (
            2 ** (z - 1)
            * gamma_spouge(0.5 * (nu + z))
            / gamma_spouge(0.5 * (2 + nu - z))
        )

    return _deriv(MK, deriv)


def Mellin_SphericalBesselJ(nu, deriv=0):
    def MK(z):
        return exp(
            log(2) * (z - 1.5) + loggamma(0.5 * (nu + z)) - loggamma(0.5 * (3 + nu - z))
        )

    return _deriv(MK, deriv)


def Mellin_FourierSine(deriv=0):
    def MK(z):
        return exp(
            log(2) * (z - 0.5) + loggamma(0.5 * (1 + z)) - loggamma(0.5 * (2 - z))
        )

    return _deriv(MK, deriv)


def Mellin_FourierCosine(deriv=0):
    def MK(z):
        return exp(log(2) * (z - 0.5) + loggamma(0.5 * z) - loggamma(0.5 * (1 - z)))

    return _deriv(MK, deriv)


def Mellin_DoubleBesselJ(alpha, nu1, nu2):
    import mpmath
    from numpy import frompyfunc

    hyp2f1 = frompyfunc(lambda *a: complex(mpmath.hyp2f1(*a)), 4, 1)
    if 0 < alpha < 1:

        def MK(z):
            return exp(
                log(2) * (z - 1)
                + log(alpha) * nu2
                + loggamma(0.5 * (nu1 + nu2 + z))
                - loggamma(0.5 * (2 + nu1 - nu2 - z))
                - loggamma(1 + nu2)
            ) * hyp2f1(
                0.5 * (-nu1 + nu2 + z), 0.5 * (nu1 + nu2 + z), 1 + nu2, alpha**2
            )

    elif alpha > 1:

        def MK(z):
            return exp(
                log(2) * (z - 1)
                + log(alpha) * (-nu1 - z)
                + loggamma(0.5 * (nu1 + nu2 + z))
                - loggamma(0.5 * (2 - nu1 + nu2 - z))
                - loggamma(1 + nu1)
            ) * hyp2f1(
                0.5 * (nu1 - nu2 + z), 0.5 * (nu1 + nu2 + z), 1 + nu1, alpha**-2
            )

    elif alpha == 1:

        def MK(z):
            return exp(
                log(2) * (z - 1)
                + loggamma(1 - z)
                + loggamma(0.5 * (nu1 + nu2 + z))
                - loggamma(0.5 * (2 + nu1 - nu2 - z))
                - loggamma(0.5 * (2 - nu1 + nu2 - z))
                - loggamma(0.5 * (2 + nu1 + nu2 - z))
            )

    else:
        raise ValueError
    return MK


def Mellin_DoubleSphericalBesselJ(alpha, nu1, nu2):
    import mpmath
    from numpy import frompyfunc

    hyp2f1 = frompyfunc(lambda *a: complex(mpmath.hyp2f1(*a)), 4, 1)
    if 0 < alpha < 1:

        def MK(z):
            return (
                pi
                * exp(
                    log(2) * (z - 3)
                    + log(alpha) * nu2
                    + loggamma(0.5 * (nu1 + nu2 + z))
                    - loggamma(0.5 * (3 + nu1 - nu2 - z))
                    - loggamma(1.5 + nu2)
                )
                * hyp2f1(
                    0.5 * (-1 - nu1 + nu2 + z),
                    0.5 * (nu1 + nu2 + z),
                    1.5 + nu2,
                    alpha**2,
                )
            )

    elif alpha > 1:

        def MK(z):
            return (
                pi
                * exp(
                    log(2) * (z - 3)
                    + log(alpha) * (-nu1 - z)
                    + loggamma(0.5 * (nu1 + nu2 + z))
                    - loggamma(0.5 * (3 - nu1 + nu2 - z))
                    - loggamma(1.5 + nu1)
                )
                * hyp2f1(
                    0.5 * (-1 + nu1 - nu2 + z),
                    0.5 * (nu1 + nu2 + z),
                    1.5 + nu1,
                    alpha**-2,
                )
            )

    elif alpha == 1:

        def MK(z):
            return pi * exp(
                log(2) * (z - 3)
                + loggamma(2 - z)
                + loggamma(0.5 * (nu1 + nu2 + z))
                - loggamma(0.5 * (3 + nu1 - nu2 - z))
                - loggamma(0.5 * (3 - nu1 + nu2 - z))
                - loggamma(0.5 * (4 + nu1 + nu2 - z))
            )

    else:
        raise ValueError
    return MK


def Mellin_Tophat(dim, deriv=0):
    def MK(z):
        return exp(
            log(2) * (z - 1)
            + loggamma(1 + 0.5 * dim)
            + loggamma(0.5 * z)
            - loggamma(0.5 * (2 + dim - z))
        )

    return _deriv(MK, deriv)


def Mellin_TophatSq(dim, deriv=0):
    if dim == 1:

        def MK(z):
            return (
                -0.25
                * sqrt(pi)
                * exp(loggamma(0.5 * (z - 2)) - loggamma(0.5 * (3 - z)))
            )

    elif dim == 3:

        def MK(z):
            return (
                2.25
                * sqrt(pi)
                * (z - 2)
                / (z - 6)
                * exp(loggamma(0.5 * (z - 4)) - loggamma(0.5 * (5 - z)))
            )

    else:

        def MK(z):
            return exp(
                log(2) * (dim - 1)
                + 2 * loggamma(1 + 0.5 * dim)
                + loggamma(0.5 * (1 + dim - z))
                + loggamma(0.5 * z)
                - loggamma(1 + dim - 0.5 * z)
                - loggamma(0.5 * (2 + dim - z))
            ) / sqrt(pi)

    return _deriv(MK, deriv)


def Mellin_Gauss(deriv=0):
    def MK(z):
        return 2 ** (0.5 * z - 1) * gamma_spouge(0.5 * z)

    return _deriv(MK, deriv)


def Mellin_GaussSq(deriv=0):
    def MK(z):
        return 0.5 * gamma_spouge(0.5 * z)

    return _deriv(MK, deriv)
