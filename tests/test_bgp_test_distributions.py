import importlib
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


class BgpTestDistributionTests(unittest.TestCase):
    def setUp(self):
        self.edges = np.array([0.8, 1.0, 1.8, 2.2, 5.0, 10.0, 12.0])
        self.rates = np.ones((len(self.edges), len(self.edges)), dtype=float)

    def test_bns_samples_stay_inside_ns_mass_range_with_boundary_clipping(self):
        sampling = importlib.import_module("bgp_test_sampling")

        mass1, mass2 = sampling.sample_bns_masses_from_grid(
            self.edges,
            self.rates,
            nsamples=2000,
            rng=np.random.default_rng(123),
            ns_mass_min=1.0,
            ns_mass_max=2.05,
        )

        self.assertEqual(mass1.shape, (2000,))
        self.assertTrue(np.all(mass1 >= mass2))
        self.assertTrue(np.all(mass1 >= 1.0))
        self.assertTrue(np.all(mass1 <= 2.05))
        self.assertTrue(np.all(mass2 >= 1.0))
        self.assertTrue(np.all(mass2 <= 2.05))

    def test_nsbh_samples_assign_bh_to_mass1_and_ns_to_mass2(self):
        sampling = importlib.import_module("bgp_test_sampling")

        mass1, mass2 = sampling.sample_nsbh_masses_from_grid(
            self.edges,
            self.rates,
            nsamples=2000,
            rng=np.random.default_rng(321),
            ns_mass_min=1.0,
            ns_mass_max=2.05,
            bh_mass_min=2.05,
            bh_mass_max=10.0,
        )

        self.assertEqual(mass1.shape, (2000,))
        self.assertTrue(np.all(mass1 >= 2.05))
        self.assertTrue(np.all(mass1 <= 10.0))
        self.assertTrue(np.all(mass2 >= 1.0))
        self.assertTrue(np.all(mass2 <= 2.05))

    def test_zero_probability_target_region_raises_value_error(self):
        sampling = importlib.import_module("bgp_test_sampling")
        rates = np.ones((len(self.edges), len(self.edges)), dtype=float)
        rates[1:4, 1:4] = 0.0

        with self.assertRaisesRegex(ValueError, "No BGP rate support"):
            sampling.sample_bns_masses_from_grid(
                self.edges,
                rates,
                nsamples=10,
                rng=np.random.default_rng(1),
                ns_mass_min=1.0,
                ns_mass_max=2.05,
            )

    def test_bns_bgp_distribution_has_expected_ranges_and_spin_limits(self):
        module = importlib.import_module("generate_bns_test_distribution")

        table = module.generate_distribution(
            nsamples=500,
            seed=10,
            mass_method="bgp",
            bgp_edges=self.edges,
            bgp_rates=self.rates,
        )

        self.assertEqual(table.colnames, ["mass1", "mass2", "spin1z", "spin2z"])
        self.assertTrue(np.all(table["mass1"] >= table["mass2"]))
        self.assertTrue(np.all(table["mass1"] <= 2.05))
        self.assertTrue(np.all(table["mass2"] >= 1.0))
        self.assertTrue(np.all(table["spin1z"] >= -0.1))
        self.assertTrue(np.all(table["spin1z"] <= 0.1))
        self.assertTrue(np.all(table["spin2z"] >= -0.1))
        self.assertTrue(np.all(table["spin2z"] <= 0.1))

    def test_nsbh_bgp_distribution_has_expected_ranges_and_spin_limits(self):
        module = importlib.import_module("generate_nsbh_test_distribution")

        table = module.generate_distribution(
            nsamples=500,
            seed=11,
            mass_method="bgp",
            bgp_edges=self.edges,
            bgp_rates=self.rates,
        )

        self.assertEqual(table.colnames, ["mass1", "mass2", "spin1z", "spin2z"])
        self.assertTrue(np.all(table["mass1"] >= 2.05))
        self.assertTrue(np.all(table["mass1"] <= 10.0))
        self.assertTrue(np.all(table["mass2"] >= 1.0))
        self.assertTrue(np.all(table["mass2"] <= 2.05))
        self.assertTrue(np.all(table["spin1z"] >= -0.99))
        self.assertTrue(np.all(table["spin1z"] <= 0.99))
        self.assertTrue(np.all(table["spin2z"] >= -0.1))
        self.assertTrue(np.all(table["spin2z"] <= 0.1))

    def test_custom_distribution_mode_remains_available(self):
        bns = importlib.import_module("generate_bns_test_distribution")
        nsbh = importlib.import_module("generate_nsbh_test_distribution")

        bns_table = bns.generate_distribution(
            nsamples=50,
            seed=12,
            mass_method="custom",
        )
        nsbh_table = nsbh.generate_distribution(
            nsamples=50,
            seed=13,
            mass_method="custom",
        )

        self.assertEqual(len(bns_table), 50)
        self.assertEqual(len(nsbh_table), 50)


if __name__ == "__main__":
    unittest.main()
