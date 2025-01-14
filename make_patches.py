import argparse

from src import PatchGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EBV-TRACER Make Patches')
    parser.add_argument('--input_svs_dir', default="", type=str)
    parser.add_argument('--save_patch_dir', default="", type=str)
    parser.add_argument('--patch_size', default=1024, type=int)
    parser.add_argument('--patch_save_size', default=512, type=int)
    parser.add_argument('--patch_step', default=1.0, type=float)
    parser.add_argument('--n_jobs', default=8, type=int)
    args = parser.parse_args()

    patch_generator = PatchGenerator(
        patch_size=args.patch_size,
        save_size=args.patch_save_size,
        patch_step=args.patch_step,
        n_jobs=args.n_jobs
    )
    patch_generator.generate_patches(
        svs_dir=args.input_svs_dir,
        base_save_dir=args.save_patch_dir,
    )
