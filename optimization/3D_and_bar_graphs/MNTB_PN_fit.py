from neuron import h
h.load_file("stdrun.hoc")

def nstomho(x, somaarea):
    return (1e-9 * x / somaarea)

class MNTB:
    def __init__(self, gid, somaarea, erev, gleak, ena, gna, gh, gka, gklt, gkht,ek,
                 cam, kam, cbm, kbm):

        self._gid = gid
        self.somaarea = somaarea
        self.erev = erev
        self.gleak = gleak
        self.ena = ena
        self.gna = gna
        self.gh = gh
        self.gka = gka
        self.gklt = gklt
        self.gkht = gkht
        self.ek = ek

        # Kinetic parameters
        self.cam = cam
        self.kam = kam
        self.cbm = cbm
        self.kbm = kbm
        self._setup_morphology()
        self._setup_biophysics()

    def _setup_morphology(self):
        self.soma = h.Section(name='soma', cell=self)
        self.soma.L = 20
        self.soma.diam = 15

    def _setup_biophysics(self):
        self.soma.Ra = 150
        self.soma.cm = 1
        self.soma.insert('leak')
        self.soma.insert('NaCh_nmb')
        self.soma.insert('IH_nmb')
        self.soma.insert('LT_dth')
        self.soma.insert('HT_dth_nmb')
        self.soma.insert('ka')

        for seg in self.soma:
            seg.leak.g = nstomho(self.gleak, self.somaarea)
            seg.leak.erev = self.erev
            seg.ena = self.ena
            seg.ek = self.ek

            seg.NaCh_nmb.gnabar = nstomho(self.gna, self.somaarea)
            seg.NaCh_nmb.cam = self.cam
            seg.NaCh_nmb.kam = self.kam
            seg.NaCh_nmb.cbm = self.cbm
            seg.NaCh_nmb.kbm = self.kbm
            seg.IH_nmb.ghbar = nstomho(self.gh, self.somaarea)
            seg.ka.gkabar = nstomho(self.gka, self.somaarea)
            seg.LT_dth.gkltbar = nstomho(self.gklt, self.somaarea)

            seg.HT_dth_nmb.gkhtbar = nstomho(self.gkht, self.somaarea)


    def __repr__(self):
        return 'MNTB [{}]'.format(self._gid)
