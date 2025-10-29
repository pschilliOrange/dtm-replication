import unittest
from thrmlDenoising.DTM import DTM
from thrmlDenoising.utils import make_cfg
from thrmlDenoising.smoke_testing import compute_conditional_accuracy, compute_free_accuracy
import jax.random as jr
import os

class TestSingleStepSavingLoading(unittest.TestCase):
    """Tests DTM saving and loading.
    
    Verifies that accuracies of a loaded model match the original after training to epoch 50 and reloading.
    """
    n_epochs = 50
    gen_batch_size = 10
    eval_key = jr.PRNGKey(42)

    @classmethod
    def setUpClass(cls):
        # DTM1: 1 step, 6_4
        cfg1 = make_cfg(
            exp=dict(seed=1, compute_autocorr=False),
            data=dict(dataset_name="smoke_testing_2_2_2", target_classes=tuple([0, 1])),
            graph=dict(graph_preset_architecture=6_4, num_label_spots=3, grayscale_levels=2, torus=False),
            sampling=dict(batch_size=2, n_samples=10, steps_per_sample=5, steps_warmup=10, training_beta=1.0),
            generation=dict(generation_beta_start=0.8, generation_beta_end=1.2, steps_warmup=20,
                            fid_images_per_digit=cls.gen_batch_size),
            diffusion_schedule=dict(num_diffusion_steps=1, kind="log", diffusion_offset=0.1),
            diffusion_rates=dict(image_rate=0.8, label_rate=0.5),
            optim=dict(momentum=0.9, b2_adam=0.999, step_learning_rates=(0.1,), n_epochs_for_lrd=70,
                       alpha_cosine_decay=0.2),
            cp=dict(adaptive_cp=False),
        )

        # Train to 50 and save every epoch so epoch_050 exists.
        cls.dtm = DTM(cfg1)
        cls.dtm.train(cls.n_epochs, evaluate_every=cls.n_epochs)
        cls.model_saving_path = cls.dtm.model_saving_path
        assert os.path.isdir(cls.model_saving_path), f"model_saving_path not found: {cls.model_saving_path}"

        # Read off accuracies at epoch 50 directly from the in-memory model.
        cls.cond_acc_50 = compute_conditional_accuracy(cls.dtm, cls.eval_key, cls.gen_batch_size)
        cls.free_acc_50 = compute_free_accuracy(cls.dtm, cls.eval_key, cls.gen_batch_size)
        print('cls.cond_acc_50, cls.free_acc_50', cls.cond_acc_50, cls.free_acc_50)

    def test_epoch50_roundtrip(self):
        # Load the checkpoint saved at epoch 50.
        loaded = DTM.load(self.model_saving_path, epoch=50)

        # Accuracies from the loaded model using the same eval key and batch size.
        cond_acc_loaded = compute_conditional_accuracy(loaded, self.eval_key, self.gen_batch_size)
        free_acc_loaded = compute_free_accuracy(loaded, self.eval_key, self.gen_batch_size)
        print('cond_acc_loaded, free_acc_loaded', cond_acc_loaded, free_acc_loaded)

        self.assertAlmostEqual(cond_acc_loaded, self.cond_acc_50, places=6,
                               msg="Loaded conditional accuracy does not match epoch-50 accuracy.")
        self.assertAlmostEqual(free_acc_loaded, self.free_acc_50, places=6,
                               msg="Loaded free accuracy does not match epoch-50 accuracy.")

class TestTwoStepStitchAndTrain(unittest.TestCase):
    """Tests stitching of two independently trained models.
    
    Verifies that accuracies of the stitched model are close to
    of the source models' accuracies after loading steps from epoch 80.
    """
    n_epochs = 80          # train the two source models to 80 epochs
    gen_batch_size = 50
    eval_key = jr.PRNGKey(123)

    @classmethod
    def setUpClass(cls):
        # Common 2-step config used by both source models (different seeds).
        common_cfg_kwargs = dict(
            data=dict(dataset_name="smoke_testing_2_2_2", target_classes=tuple([0, 1])),
            graph=dict(graph_preset_architecture=6_4, num_label_spots=3, grayscale_levels=2, torus=False),
            sampling=dict(batch_size=2, n_samples=10, steps_per_sample=5, steps_warmup=10, training_beta=1.0),
            generation=dict(generation_beta_start=0.8, generation_beta_end=1.2, steps_warmup=20,
                            fid_images_per_digit=cls.gen_batch_size),
            diffusion_schedule=dict(num_diffusion_steps=2, kind="log", diffusion_offset=0.1),
            diffusion_rates=dict(image_rate=0.8, label_rate=0.5),
            optim=dict(momentum=0.9, b2_adam=0.999, step_learning_rates=(0.1, 0.1),
                       n_epochs_for_lrd=70, alpha_cosine_decay=0.2),
            cp=dict(adaptive_cp=False),
            exp=dict(compute_autocorr=False),
        )

        cfg1 = make_cfg(**{**common_cfg_kwargs, "exp": dict(seed=1, compute_autocorr=False)})
        cfg2 = make_cfg(**{**common_cfg_kwargs, "exp": dict(seed=10, compute_autocorr=False)})

        # Train two independent 2-step DTMs to 80 epochs each.
        cls.dtm1 = DTM(cfg1)
        cls.dtm1.train(cls.n_epochs, evaluate_every=cls.n_epochs)
        cls.model_saving_path1 = cls.dtm1.model_saving_path
        assert os.path.isdir(cls.model_saving_path1), f"model_saving_path1 not found: {cls.model_saving_path1}"

        cls.dtm2 = DTM(cfg2)
        cls.dtm2.train(cls.n_epochs, evaluate_every=cls.n_epochs)
        cls.model_saving_path2 = cls.dtm2.model_saving_path
        assert os.path.isdir(cls.model_saving_path2), f"model_saving_path2 not found: {cls.model_saving_path2}"

        # Stitch: take step 0 from run1 @ epoch 80 and step 1 from run2 @ epoch 80.
        step_sources = [
            os.path.join(cls.model_saving_path1, f"epoch_{cls.n_epochs:03d}", "step_00.eqx"),
            os.path.join(cls.model_saving_path2, f"epoch_{cls.n_epochs:03d}", "step_01.eqx"),
        ]
        for p in step_sources:
            assert os.path.isfile(p), f"Missing step file: {p}"

        # Use run1's base config for structure; provide per-step sources from both runs.
        cls.stitched = DTM.load(base_path=cls.model_saving_path1, step_sources=step_sources)

        cls.cond_acc1 = compute_conditional_accuracy(cls.dtm1, cls.eval_key, cls.gen_batch_size)
        cls.free_acc1 = compute_free_accuracy(cls.dtm1, cls.eval_key, cls.gen_batch_size)
        cls.cond_acc2 = compute_conditional_accuracy(cls.dtm2, cls.eval_key, cls.gen_batch_size)
        cls.free_acc2 = compute_free_accuracy(cls.dtm2, cls.eval_key, cls.gen_batch_size)
        cls.cond_acc_stitched = compute_conditional_accuracy(cls.stitched, cls.eval_key, cls.gen_batch_size)
        cls.free_acc_stitched = compute_free_accuracy(cls.stitched, cls.eval_key, cls.gen_batch_size)
        print("Model accuracies:",
              "dtm1:", float(cls.cond_acc1), float(cls.free_acc1),
              "dtm2:", float(cls.cond_acc2), float(cls.free_acc2),
              "stitched:", float(cls.cond_acc_stitched), float(cls.free_acc_stitched))

    def test_stitched_model_performance(self):
        conditional_tolerance = 5.0
        free_tolerance = 30.0
        self.assertAlmostEqual(self.cond_acc_stitched, self.cond_acc1, delta=conditional_tolerance,
                               msg="Stitched conditional accuracy deviates >5% from dtm1.")
        self.assertAlmostEqual(self.cond_acc_stitched, self.cond_acc2, delta=conditional_tolerance,
                               msg="Stitched conditional accuracy deviates >5% from dtm2.")
        self.assertAlmostEqual(self.free_acc_stitched, self.free_acc1, delta=free_tolerance,
                               msg="Stitched free accuracy deviates >30% from dtm1.")
        self.assertAlmostEqual(self.free_acc_stitched, self.free_acc2, delta=free_tolerance,
                               msg="Stitched free accuracy deviates >30% from dtm2.")
        
if __name__ == "__main__":
    unittest.main(verbosity=2)