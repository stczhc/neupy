
! atomic compare: compare two structures
! find the d, given dmax as a upper limit
! when used in filtering, this will be accurate
! this is very efficient, if dmax is not large
SUBROUTINE at_comp(n, x, y, nele, eles, dmax, d, msel)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, nele
  REAL(8), INTENT(IN) :: x(n, 3), y(n, 3), dmax
  INTEGER, INTENT(IN) :: eles(n) ! must start from 1
  REAL(8), INTENT(OUT) :: d
  INTEGER, INTENT(OUT) :: msel(n)
  
  INTEGER :: nea(nele), ndi(nele, n), ndu(nele, n)
  INTEGER :: sel(n), est(n), esti
  INTEGER :: i, j, ai, si, ei
  REAL(8) :: dl(n), dtmp, dmin, dh, dy(n, n)
  
  nea = 0
  ndi = 0
  DO i = 1, n
    nea(eles(i)) = nea(eles(i)) + 1
    ndi(eles(i), nea(eles(i))) = i
  END DO
  DO i = 1, n
    DO j = 1, i - 1
      dy(j, i) = SQRT(SUM((y(j, :) - y(i, :)) ** 2))
    END DO
  END DO
  ndu = ndi
  ai = 1
  sel(1) = 0
  dmin = dmax
  esti = 0
  msel = 0
  DO WHILE (.TRUE.)
    ei = eles(ai)
    si = 0
    DO i = 1, nea(ei)
      IF (ndu(ei, i) > sel(ai)) THEN
        si = ndu(ei, i)
        esti = esti + 1
        est(esti) = i
        ndu(ei, i) = 0
        EXIT
      END IF
    END DO
    IF (si == 0) THEN
      IF (ai /= 1) THEN
        ai = ai - 1
        ndu(ei, est(esti)) = ndi(ei, est(esti))
        esti = esti - 1
        CYCLE
      ELSE
        EXIT
      END IF
    END IF
    sel(ai) = si
    IF (ai == 1) THEN
      dl(ai) = 0
    ELSE
      dl(ai) = dl(ai - 1)
    END IF
    DO i = 1, ai - 1
      dh = ABS(SQRT(SUM((x(sel(ai), :) - x(sel(i), :)) ** 2)) - dy(i, ai))
      IF (dh > dl(ai)) dl(ai) = dh
    END DO
    IF (dl(ai) > dmin .OR. ai == n) THEN
      IF (ai == n .AND. dl(ai) < dmin) THEN
        dmin = dl(ai)
        msel = sel
      END IF
      ndu(ei, est(esti)) = ndi(ei, est(esti))
      esti = esti - 1
    ELSE
      ai = ai + 1
      IF (ai <= n) sel(ai) = 0
    END IF
  END DO
  d = dmin
END SUBROUTINE

! atomic sort: sort lengths to find the sequence
! with smallest inverse-order sum
! when used in filtering, this will give approximate results
! but this will only state similar structure to be diff
SUBROUTINE at_sort(n, x, nele, eles, l)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, nele
  REAL(8), INTENT(IN) :: x(n, 3)
  INTEGER, INTENT(IN) :: eles(n) ! must start from 1
  REAL(8), INTENT(OUT) :: l(n * (n - 1) / 2)
  
  INTEGER :: nea(nele), ndi(nele, n), ndu(nele, n)
  INTEGER :: sel(n), est(n), esti
  INTEGER :: i, li, ai, si, ei
  REAL(8) :: ltmp(n * (n - 1) / 2), rtmp, rmin
  
  nea = 0
  ndi = 0
  DO i = 1, n
    nea(eles(i)) = nea(eles(i)) + 1
    ndi(eles(i), nea(eles(i))) = i
  END DO
  ndu = ndi
  ai = 1
  li = 1
  sel(1) = 0
  rtmp = 0D0
  rmin = 999D0
  esti = 0
  DO WHILE (.TRUE.)
    ei = eles(ai)
    si = 0
    DO i = 1, nea(ei)
      IF (ndu(ei, i) > sel(ai)) THEN
        si = ndu(ei, i)
        esti = esti + 1
        est(esti) = i
        ndu(ei, i) = 0
        EXIT
      END IF
    END DO
    IF (si == 0) THEN
      IF (ai /= 1) THEN
        ai = ai - 1
        DO i = 1, ai - 1
          li = li - 1
          IF (li /= 1) THEN
            IF (ltmp(li) < ltmp(li - 1)) rtmp = rtmp - (ltmp(li - 1) - ltmp(li))
          END IF
        END DO
        ndu(ei, est(esti)) = ndi(ei, est(esti))
        esti = esti - 1
        CYCLE
      ELSE
        EXIT
      END IF
    END IF
    sel(ai) = si
    DO i = 1, ai - 1
      ltmp(li) = SQRT(SUM((x(sel(ai), :) - x(sel(i), :)) ** 2))
      IF (li /= 1) THEN
        IF (ltmp(li) < ltmp(li - 1)) rtmp = rtmp + ltmp(li - 1) - ltmp(li)
      END IF
      li = li + 1
    END DO
    IF (rtmp > rmin .OR. ai == n) THEN
      IF (ai == n .AND. rtmp < rmin) THEN
        rmin = rtmp
        l = ltmp
      END IF
      DO i = 1, ai - 1
        li = li - 1
        IF (li /= 1) THEN
          IF (ltmp(li) < ltmp(li - 1)) rtmp = rtmp - (ltmp(li - 1) - ltmp(li))
        END IF
      END DO
      ndu(ei, est(esti)) = ndi(ei, est(esti))
      esti = esti - 1
    ELSE
      ai = ai + 1
      IF (ai <= n) sel(ai) = 0
    END IF
  END DO
END SUBROUTINE