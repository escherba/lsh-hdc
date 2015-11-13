! F2PY experiment
! http://stackoverflow.com/a/12200671/597371

subroutine minmaxr(a,n,amin,amax)
    implicit none
    !f2py intent(hidden) :: n
    !f2py intent(out) :: amin,amax
    !f2py intent(in) :: a
    integer n
    real a(n),amin,amax,acurr
    integer i
    real :: x = 0

    if(n > 0)then
        amin = a(1)
        amax = a(1)
        do i=2, n
            acurr = a(i)
            if(acurr > amax)then
                amax = acurr
            elseif(acurr < amin) then
                amin = acurr
            endif
        enddo
    else
        ! set return values to (inf, -inf)
        amin = -log(x)
        amax = -amin
    endif
end subroutine minmaxr
