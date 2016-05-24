
! qsd eval
SUBROUTINE xqsd_eval(x, e, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: x(0:n-1)
  REAL(8), INTENT(OUT) :: e
  INTEGER :: i
  REAL(8), PARAMETER :: a0(0:3) = (/ -200d0, -100d0, -170d0, 15d0 /)
  REAL(8), PARAMETER :: a(0:3) = (/ -1d0, -1d0, -6.5d0, 0.7d0 /)
  REAL(8), PARAMETER :: b(0:3) = (/ 0d0, 0d0, 11d0, 0.6d0 /)
  REAL(8), PARAMETER :: c(0:3) = (/ -10d0, -10d0, -6.5d0, 0.7d0 /)
  REAL(8), PARAMETER :: x0(0:3) = (/ 1d0, 0d0, -0.5d0, -1d0 /)
  REAL(8), PARAMETER :: y0(0:3) = (/ 0d0, 0.5d0, 1.5d0, 1d0 /)

  e = 0d0
  DO i = 0, 3
    e = e + a0(i) * EXP(a(i)*(x(0)-x0(i))**2 + b(i)*(x(0)-x0(i))*(x(1)-y0(i)) + &
      c(i)*(x(1)-y0(i))**2)
  END DO

END SUBROUTINE

! qsd evald
SUBROUTINE xqsd_evald(x, e, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: x(0:n-1)
  REAL(8), INTENT(OUT) :: e(0:n-1)
  INTEGER :: i
  REAL(8), PARAMETER :: a0(0:3) = (/ -200d0, -100d0, -170d0, 15d0 /)
  REAL(8), PARAMETER :: a(0:3) = (/ -1d0, -1d0, -6.5d0, 0.7d0 /)
  REAL(8), PARAMETER :: b(0:3) = (/ 0d0, 0d0, 11d0, 0.6d0 /)
  REAL(8), PARAMETER :: c(0:3) = (/ -10d0, -10d0, -6.5d0, 0.7d0 /)
  REAL(8), PARAMETER :: x0(0:3) = (/ 1d0, 0d0, -0.5d0, -1d0 /)
  REAL(8), PARAMETER :: y0(0:3) = (/ 0d0, 0.5d0, 1.5d0, 1d0 /)

  e = 0d0
  DO i = 0, 3
    e(0) = e(0) + a0(i) * EXP(a(i)*(x(0)-x0(i))**2 + b(i)*(x(0)-x0(i))*(x(1)-y0(i)) + &
      c(i)*(x(1)-y0(i))**2) * (2*a(i)*(x(0)-x0(i)) + b(i)*(x(1)-y0(i)))
    e(1) = e(1) + a0(i) * EXP(a(i)*(x(0)-x0(i))**2 + b(i)*(x(0)-x0(i))*(x(1)-y0(i)) + &
      c(i)*(x(1)-y0(i))**2) * (2*c(i)*(x(1)-y0(i)) + b(i)*(x(0)-x0(i)))
  END DO

END SUBROUTINE

! qsd evaldd
SUBROUTINE xqsd_evaldd(x, h, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: x(0:n-1)
  REAL(8), INTENT(OUT) :: h(0:n-1,0:n-1)
  REAL(8) :: e(0:2)
  INTEGER :: i
  REAL(8), PARAMETER :: a0(0:3) = (/ -200d0, -100d0, -170d0, 15d0 /)
  REAL(8), PARAMETER :: a(0:3) = (/ -1d0, -1d0, -6.5d0, 0.7d0 /)
  REAL(8), PARAMETER :: b(0:3) = (/ 0d0, 0d0, 11d0, 0.6d0 /)
  REAL(8), PARAMETER :: c(0:3) = (/ -10d0, -10d0, -6.5d0, 0.7d0 /)
  REAL(8), PARAMETER :: x0(0:3) = (/ 1d0, 0d0, -0.5d0, -1d0 /)
  REAL(8), PARAMETER :: y0(0:3) = (/ 0d0, 0.5d0, 1.5d0, 1d0 /)

  e = 0d0
  DO i = 0, 3
    e(0) = e(0) + a0(i) * EXP(a(i)*(x(0)-x0(i))**2 + b(i)*(x(0)-x0(i))*(x(1)-y0(i)) + &
      c(i)*(x(1)-y0(i))**2) * 2*a(i) + a0(i) * EXP(a(i)*(x(0)-x0(i))**2 + &
      b(i)*(x(0)-x0(i))*(x(1)-y0(i)) + &
      c(i)*(x(1)-y0(i))**2) * (2*a(i)*(x(0)-x0(i)) + b(i)*(x(1)-y0(i)))**2
    e(1) = e(1) + a0(i) * EXP(a(i)*(x(0)-x0(i))**2 + b(i)*(x(0)-x0(i))*(x(1)-y0(i)) + &
      c(i)*(x(1)-y0(i))**2) * b(i) + a0(i) * EXP(a(i)*(x(0)-x0(i))**2 + &
      b(i)*(x(0)-x0(i))*(x(1)-y0(i)) + &
      c(i)*(x(1)-y0(i))**2) * (2*c(i)*(x(1)-y0(i)) + b(i)*(x(0)-x0(i))) * &
      (2*a(i)*(x(0)-x0(i))+ b(i)*(x(1)-y0(i)))
    e(2) = e(2) + a0(i) * EXP(a(i)*(x(0)-x0(i))**2 + b(i)*(x(0)-x0(i))*(x(1)-y0(i)) + &
      c(i)*(x(1)-y0(i))**2) * 2*c(i) + a0(i) * EXP(a(i)*(x(0)-x0(i))**2 + &
      b(i)*(x(0)-x0(i))*(x(1)-y0(i)) + &
      c(i)*(x(1)-y0(i))**2) * (2*c(i)*(x(1)-y0(i)) +b(i)*(x(0)-x0(i)))**2
  END DO
  h(0, 0) = e(0)
  h(0, 1) = e(1)
  h(1, 0) = e(1)
  h(1, 1) = e(2)
END SUBROUTINE xqsd_evaldd
