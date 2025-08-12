import h5py, pandas as pd

for fn in [r'E:\generate_mixture\gas_hdf5\no.hdf5',
           r'E:\generate_mixture\gas_hdf5\no2.hdf5']:
    with h5py.File(fn, 'r') as f:
        arr = f['lbl'][:]
    df = pd.DataFrame({c: arr[c] for c in arr.dtype.names})
    ids = sorted(df['molec_id'].unique())
    inwin = df['nu'].between(400, 4000)
    print('\nFILE:', fn)
    print('  Molecule IDs present:', ids)
    print('  ν-range:', float(df['nu'].min()), '→', float(df['nu'].max()))
    for m in [8, 10]:
        print(f'  lines for molec_id={m} in 400–4000 cm^-1 :',
              int(((df['molec_id']==m) & inwin).sum()))
