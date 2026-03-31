TITLE Kv7 model adapted from 
: Adaptation of Kv7 model from Arne Battefeld et al., 2014 

NEURON {
	SUFFIX kv7
	USEION k READ ek WRITE ik
    RANGE  gbar, i
    : GLOBAL ninf, tau_n
}

UNITS {
    (S) = (siemens)
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {	
	gbar = .0001 (S/cm2)
	Ca = 0.044 (/ms) : Ca = 0.036 (/ms) old params
	Cb = 0.0011 (/ms) : Cb = 0.002 (/ms)
	za = 1.4991 (mV) : za = 0.909 (mV)
	zb = 1.6921 (mV) : zb = 1.102 (mV)
	F_RT = 26.55 (mV)
}

ASSIGNED {
	v (mV) : NEURON provides this
	ek (mV)
	ik (mA/cm2)
    i (mA/cm2)
	tau_n (ms)
    ninf
}

STATE { n }

INITIAL {
	rates(v)
    : Assume that equilibrium has been reached
	n = alpha(v)/(alpha(v)+beta(v))
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	i = gbar*n*(v-ek)
    ik = i
}

DERIVATIVE state {
    rates(v)
	n' = (ninf - n)/tau_n
}

FUNCTION alpha(v(mV)) {
    alpha = Ca * exp(za * v * F_RT / 1000) / 3: Divide by a thousand because activation curves are in V
}

FUNCTION beta(v(mV)) {
    beta = Cb * exp(- zb * v * F_RT / 1000) / 3 : Divide by a thousand because activation curves are in V
}

FUNCTION rates(v(mV)) (/ms) {
	tau_n = 1.0 / (alpha(v) + beta(v))
	ninf = alpha(v) * tau_n
}
