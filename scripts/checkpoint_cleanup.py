"""Explicit checkpoint maintenance. Dry-run by default; never deletes outside root."""
import argparse,time
from pathlib import Path

def expired_checkpoints(root, days=7, now=None):
    root=Path(root).resolve()
    if days < 1:raise ValueError('Retention must be at least one day')
    cutoff=(time.time() if now is None else now)-days*86400
    return [p for p in root.glob('*.json') if not p.is_symlink() and p.resolve().parent==root and p.stat().st_mtime<cutoff]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--days',type=int,default=7);ap.add_argument('--delete',action='store_true');args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]/'v2_output/react/checkpoints'
    files=expired_checkpoints(root,args.days)
    for p in files:
        print(p.name)
        if args.delete:p.unlink()
    print(f'{len(files)} expired checkpoints; '+('deleted' if args.delete else 'dry run'))
if __name__=='__main__':main()
