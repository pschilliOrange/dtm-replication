import unittest
from thrmlDenoising.step import DiffusionStep
from thrmlDenoising.DTM import DTM
from thrmlDenoising.utils import make_cfg
from thrmlDenoising.smoke_testing import compute_conditional_accuracy, compute_free_accuracy
import jax.random as jr

class SingleStepGrid6_4(unittest.TestCase):
    n_epochs: int = 100
    gen_batch_size: int = 10
    key = jr.PRNGKey(0)

    @classmethod
    def setUpClass(cls):
        cls.dtm = cls._build_trained_dtm()

    @classmethod
    def _build_trained_dtm(cls) -> DTM:
        cfg = make_cfg(
            exp=dict(seed=1, compute_autocorr=False),
            data=dict(dataset_name="smoke_testing_2_2_2", target_classes=tuple([0, 1])),
            graph=dict(graph_preset_architecture=6_4, num_label_spots=3, grayscale_levels=2, torus=False, base_graph_manager = "poisson_binomial_ising_graph_manager"),
            sampling=dict(batch_size=2, n_samples=10, steps_per_sample=5, steps_warmup=10, training_beta=1.0),
            generation=dict(generation_beta_start=0.8, generation_beta_end=1.2, steps_warmup=20, fid_images_per_digit=10),
            diffusion_schedule=dict(num_diffusion_steps=1, kind="log", diffusion_offset=0.1),
            diffusion_rates=dict(image_rate=0.8, label_rate=0.5),
            optim=dict(momentum=0.9, b2_adam=0.999, step_learning_rates=(0.1,), n_epochs_for_lrd=70, alpha_cosine_decay=0.2),
            cp=dict(adaptive_cp=False),
        )
        dtm = DTM(cfg)
        dtm.train(cls.n_epochs, 1)
        return dtm

    def test_conditional_generation_accuracy(self):
        cond_pct = compute_conditional_accuracy(self.dtm, self.key, self.gen_batch_size)
        print(f"Conditional accuracy {cond_pct:.2f}%")
        self.assertGreaterEqual(cond_pct, 80.0, msg=f"Conditional accuracy {cond_pct:.2f}% < 80%")

    def test_free_generation_accuracy(self):
        free_pct = compute_free_accuracy(self.dtm, self.key, self.gen_batch_size)
        print(f"Free accuracy {free_pct:.2f}%")
        self.assertGreaterEqual(free_pct, 60.0, msg=f"Free accuracy {free_pct:.2f}% < 60%")

class TwoStepsGrid6_4(unittest.TestCase):
    n_epochs: int = 100
    gen_batch_size: int = 10
    key = jr.PRNGKey(0)

    @classmethod
    def setUpClass(cls):
        cls.dtm = cls._build_trained_dtm()

    @classmethod
    def _build_trained_dtm(cls) -> DTM:
        cfg = make_cfg(
            exp=dict(seed=1, compute_autocorr=False),
            data=dict(dataset_name="smoke_testing_3_1_3", target_classes=tuple([0, 1, 2])),
            graph=dict(graph_preset_architecture=8_8, num_label_spots=2, grayscale_levels=1, torus=False),
            sampling=dict(batch_size=2, n_samples=10, steps_per_sample=5, steps_warmup=20, training_beta=1.0),
            generation=dict(generation_beta_start=0.8, generation_beta_end=1.2, steps_warmup=40, fid_images_per_digit=cls.gen_batch_size),
            diffusion_schedule=dict(num_diffusion_steps=2, kind="log", diffusion_offset=0.1),
            diffusion_rates=dict(image_rate=0.8, label_rate=0.5),
            optim=dict(momentum=0.9, b2_adam=0.999, step_learning_rates=(0.1,), n_epochs_for_lrd=70, alpha_cosine_decay=0.2),
            cp=dict(adaptive_cp=False),
        )
        dtm = DTM(cfg)
        dtm.train(cls.n_epochs, 1)
        return dtm

    def test_conditional_generation_accuracy(self):
        cond_pct = compute_conditional_accuracy(self.dtm, self.key, self.gen_batch_size)
        print(f"Conditional accuracy {cond_pct:.2f}%")
        self.assertGreaterEqual(cond_pct, 80.0, msg=f"Conditional accuracy {cond_pct:.2f}% < 80%")

    def test_free_generation_accuracy(self):
        free_pct = compute_free_accuracy(self.dtm, self.key, self.gen_batch_size)
        print(f"Free accuracy {free_pct:.2f}%")
        self.assertGreaterEqual(free_pct, 60.0, msg=f"Free accuracy {free_pct:.2f}% < 60%")


if __name__ == "__main__":
    unittest.main(verbosity=2)