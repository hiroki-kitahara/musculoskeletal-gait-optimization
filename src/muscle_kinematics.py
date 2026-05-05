import casadi as ca

class PolynomialMuscleKinematics:

  def __init__(self, spanned_coordinate_indices_list, exponents_lists, coefficients_list):
    self.spanned_coordinate_indices_list = spanned_coordinate_indices_list
    self.exponents_lists = exponents_lists
    self.coefficients_list = coefficients_list

  def get_kinematics(self, q, v):
    lMT = []

    for spanned_coordinate_indices, exponents_list, coefficients in zip(self.spanned_coordinate_indices_list, self.exponents_lists, self.coefficients_list):
      monomials = []

      for exponents in exponents_list:
        monomial = 1.0
        for spanned_coordinate_index, exponent in zip(spanned_coordinate_indices, exponents):
          monomial *= q[spanned_coordinate_index] ** exponent

        monomials.append(monomial)

      lMT.append(ca.DM(coefficients).T @ ca.vertcat(*monomials))

    lMT = ca.vertcat(*lMT)

    MA = -ca.jacobian(lMT, q).T

    vMT = -MA.T @ v

    return lMT, vMT, MA
