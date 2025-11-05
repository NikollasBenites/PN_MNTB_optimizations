: 	Hyperpolarization activated current (Ih) Sierksma et al., (2017)
:	and Wang et al., (1998)


NEURON {
	SUFFIX IH_nmb
	NONSPECIFIC_CURRENT i
    RANGE ghbar, gh, ih
	RANGE cau, kau, cbu, kbu

}


UNITS {
	(mV) = (millivolt)
	(S) = (mho)
	(mA) = (milliamp)
}

PARAMETER {
	v (mV)
	ghbar = .0037 (S/cm2)
	eh =  -45.0 (mV)
	q10tau = 3.0
	q10g = 2.0

	cau = 9.12e-8 (/ms)
	kau = -0.1 (/mV)
	cbu = .0021 (/ms)
	kbu = 0 (/mV)

}

ASSIGNED {
	celsius (degC)
	gh (S/cm2)
	i (mA/cm2)

	uinf
	utau (ms)
	qg ()  : computed q10 for ghbar based on q10g
    q10 ()

	au (/ms)
	bu (/ms)
}

STATE {
	u
}

INITIAL {
	qg = q10g^((celsius-22)/10 (degC))
    q10 = q10tau^((celsius - 22)/10 (degC)) : if you don't like room temp, it can be changed!
	rates(v)
	u = uinf
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	gh = qg*ghbar*u
	i = gh*(v - eh)
}

DERIVATIVE state {
	rates(v)
	u' = (uinf - u)/utau
}

PROCEDURE rates(v(mV)) {
	LOCAL au, bu

	au = cau*exp(kau*v)
	bu = cbu*exp(kbu*v)

	uinf = au/(au + bu)
	utau = 1/(au + bu)
	utau = utau/q10

}

