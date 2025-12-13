import numpy as np

from modflow_devtools.netcdf import (
    FILL_DNODATA,
    FILL_FLOAT64,
    FILL_INT64,
    NetCDFModel,
    NetCDFPackage,
    NetCDFParam,
)


def test_om_model():
    packages = [
        {
            "package_name": "welg_0",
            "package_type": "gwf-wElg",
            "params": [
                {
                    "name": "aux",
                    "attrs": {"modflow_input": "GWFMODEL/WELG0/AUX", "modflow_iaux": 1},
                    "encodings": {"_FillValue": 3e30},
                    "shape": ["time", "z", "y", "x"],
                    "dtype": "float64",
                    # "data": optional <array> in shape,
                },
                {
                    "name": "q",
                    "attrs": {"modflow_input": "GWFMODEL/WELG0/Q"},
                    "encodings": {"_FillValue": 3e30},
                    "shape": ["time", "z", "y", "x"],
                    "dtype": "float64",
                },
            ],
        },
    ]
    attrs = {
        "modflow_grid": "structured",
        "modflow_model": "gwf6: gwFmodel",
    }
    nc_meta = {
        "attrs": attrs,
        "packages": packages,
    }

    # classmethod to generate and validate model
    inst = NetCDFModel.from_meta(nc_meta, context={"dims": [2, 4, 3, 2]})

    # meta dict from model instance
    print(inst.meta)

    # dataset from model instance
    ds = inst.to_xarray()
    print(ds)

    # regenerate new validated instance from earlier dataset
    # inst = NetCDFModel.from_xarray(ds)
    # print(inst.meta)

    assert ds.attrs["modflow_grid"] == "structured"
    assert ds.attrs["modflow_model"] == "gwf6: gwfmodel"
    assert "mesh" not in ds.attrs
    assert "welg_0_q" in ds
    assert "welg_0_aux" in ds
    assert np.allclose(ds["welg_0_q"].values, FILL_DNODATA)
    assert np.allclose(ds["welg_0_aux"].values, FILL_DNODATA)
    assert ds["welg_0_q"].dims == ("time", "z", "y", "x")
    assert ds["welg_0_aux"].dims == ("time", "z", "y", "x")
    assert ds.sizes["time"] == 2
    assert ds.sizes["z"] == 4
    assert ds.sizes["y"] == 3
    assert ds.sizes["x"] == 2
    assert len(ds) == 2

    for p in packages:
        # TODO: define required context for isolated package
        pinst = NetCDFPackage.model_validate(p, context={"modelname": "gwfmodel"})
        print(pinst.meta)


def test_om_package():
    packages = [
        {
            "package_name": "welg_0",
            "package_type": "gwf-welg",
            "params": [
                {
                    "name": "aux",
                },
                {
                    "name": "q",
                },
            ],
        },
    ]

    context = {"modelname": "gwfmodel", "grid": "structured", "dims": [1, 1, 1, 1]}
    for p in packages:
        pinst = NetCDFPackage.from_meta(p, context=context)
        ds = pinst.to_xarray()
        print(ds)
        meta = pinst.meta
        print(meta)
        pinst = NetCDFPackage.model_validate(meta, context=context)


def test_om_param():
    params = [
        {
            "name": "aux",
        },
        {
            "name": "q",
        },
    ]
    context = {
        "modelname": "gwfmodel",
        "grid": "structured",
        "package_name": "welg0",
        "package_type": "gwf-welg",
        "dims": [1, 1, 1, 1],
    }

    for p in params:
        # pinst = NetCDFParam.model_validate(p, context=context)
        pinst = NetCDFParam.from_meta(p, context=context)
        print(pinst.meta)
        ds = pinst.to_xarray()
        print(ds)


def test_om_model_mesh():
    packages = [
        {
            "package_name": "dis",
            "package_type": "gwf-dis",
            "params": [
                {
                    "name": "delr",
                },
                {
                    "name": "delc",
                },
                {
                    "name": "idomain",
                },
            ],
        },
        {
            "package_name": "npf",
            "package_type": "gwf-npf",
            "params": [
                {
                    "name": "icelltype",
                },
                {
                    "name": "k",
                },
                {
                    "name": "k22",
                },
            ],
        },
        {
            "package_name": "welg_0",
            "package_type": "gwf-welg",
            "params": [
                {
                    "name": "aux",
                    "attrs": {"modflow_input": "GWFMODEL/WELG0/AUX", "modflow_iaux": 1},
                    "encodings": {"_FillValue": 3e30},
                    "shape": ["time", "nmesh_face"],
                    "dtype": "float64",
                    # "data": optional <array> in shape,
                },
                {
                    "name": "q",
                    "attrs": {"modflow_input": "GWFMODEL/WELG0/Q"},
                    "encodings": {"_FillValue": 3e30},
                    "shape": ["time", "nmesh_face"],
                    "dtype": "float64",
                },
            ],
        },
    ]
    attrs = {
        "modflow_grid": "structured",
        "modflow_model": "gwf6: gwfmodel",
        "mesh": "layered",
    }
    nc_meta = {
        "attrs": attrs,
        "packages": packages,
    }

    # classmethod to generate and validate model
    inst = NetCDFModel.from_meta(nc_meta, context={"dims": [2, 3, 2, 2]})

    # meta dict from model instance
    # meta1 = inst.meta

    # dataset from model instance
    ds = inst.to_xarray()
    print(ds)

    # regenerate new validated instance from earlier dataset
    # inst = NetCDFModel.from_xarray(ds)
    # print(inst.meta)
    # meta2 = inst.meta

    # assert meta1 == meta2
    # print(meta1)
    # print(meta2)
    # np.testing.assert_equal(meta1, meta2)

    assert ds.attrs["modflow_grid"] == "structured"
    assert ds.attrs["modflow_model"] == "gwf6: gwfmodel"
    assert ds.attrs["mesh"] == "layered"
    assert "dis_delr" in ds
    assert "dis_delc" in ds
    assert np.allclose(ds["dis_delr"].values, FILL_FLOAT64)
    assert np.allclose(ds["dis_delc"].values, FILL_FLOAT64)
    assert ds["dis_delr"].dims == ("x",)
    assert ds["dis_delc"].dims == ("y",)
    for layer in range(3):
        assert f"dis_idomain_l{layer + 1}" in ds
        assert f"npf_k_l{layer + 1}" in ds
        assert f"npf_k22_l{layer + 1}" in ds
        assert f"npf_icelltype_l{layer + 1}" in ds
        assert f"welg_0_q_l{layer + 1}" in ds
        assert f"welg_0_aux_l{layer + 1}" in ds
        assert np.allclose(ds[f"dis_idomain_l{layer + 1}"].values, FILL_INT64)
        assert np.allclose(ds[f"npf_k_l{layer + 1}"].values, FILL_FLOAT64)
        assert np.allclose(ds[f"npf_k22_l{layer + 1}"].values, FILL_FLOAT64)
        assert np.allclose(ds[f"npf_icelltype_l{layer + 1}"].values, FILL_INT64)
        assert np.allclose(ds[f"welg_0_q_l{layer + 1}"].values, FILL_DNODATA)
        assert np.allclose(ds[f"welg_0_aux_l{layer + 1}"].values, FILL_DNODATA)
        assert ds[f"dis_idomain_l{layer + 1}"].dims == ("nmesh_face",)
        assert ds[f"npf_k_l{layer + 1}"].dims == ("nmesh_face",)
        assert ds[f"npf_k22_l{layer + 1}"].dims == ("nmesh_face",)
        assert ds[f"npf_icelltype_l{layer + 1}"].dims == ("nmesh_face",)
        assert ds[f"welg_0_q_l{layer + 1}"].dims == ("time", "nmesh_face")
        assert ds[f"welg_0_aux_l{layer + 1}"].dims == ("time", "nmesh_face")
    assert ds.sizes["time"] == 2
    assert ds.sizes["nmesh_face"] == 4
    assert ds.sizes["x"] == 2
    assert ds.sizes["y"] == 2
    assert len(ds) == 20

    # for p in packages:
    # TODO: define required context for isolated package
    #    pinst = NetCDFPackage.model_validate(p, context={"modelname": "gwfmodel"})
    #    print(pinst.meta)


def test_om_package_mesh():
    packages = [
        {
            "package_name": "npf",
            "package_type": "gwf-npf",
            "params": [
                {
                    "name": "icelltype",
                },
                {
                    "name": "k",
                },
                {
                    "name": "k22",
                },
            ],
        },
        {
            "package_name": "welg_0",
            "package_type": "gwf-welg",
            "params": [
                {
                    "name": "aux",
                },
                {
                    "name": "q",
                },
            ],
        },
    ]

    context = {
        "mesh": "layered",
        "modelname": "gwfmodel",
        "grid": "structured",
        "dims": [2, 3, 2, 2],
    }
    for p in packages:
        pinst = NetCDFPackage.from_meta(p, context=context)
        ds = pinst.to_xarray()
        print(ds)
        meta = pinst.meta
        # print(meta)
        pinst = NetCDFPackage.model_validate(meta, context=context)

        print(ds.attrs)

        if p["package_type"] == "gwf-npf":
            # TODO: still write model attrs?
            assert len(ds.attrs) == 0
            for layer in range(3):
                assert f"npf_k_l{layer + 1}" in ds
                assert f"npf_k22_l{layer + 1}" in ds
                assert f"npf_icelltype_l{layer + 1}" in ds
                assert np.allclose(ds[f"npf_k_l{layer + 1}"].values, FILL_FLOAT64)
                assert np.allclose(ds[f"npf_k22_l{layer + 1}"].values, FILL_FLOAT64)
                assert np.allclose(ds[f"npf_icelltype_l{layer + 1}"].values, FILL_INT64)
                assert ds[f"npf_k_l{layer + 1}"].dims == ("nmesh_face",)
                assert ds[f"npf_k22_l{layer + 1}"].dims == ("nmesh_face",)
                assert ds[f"npf_icelltype_l{layer + 1}"].dims == ("nmesh_face",)
            assert ds.sizes["nmesh_face"] == 4
            assert len(ds) == 9

        elif p["package_type"] == "gwf-welg":
            # TODO: still write model attrs?
            assert len(ds.attrs) == 0
            for layer in range(3):
                assert f"welg_0_q_l{layer + 1}" in ds
                assert f"welg_0_aux_l{layer + 1}" in ds
                assert np.allclose(ds[f"welg_0_q_l{layer + 1}"].values, FILL_DNODATA)
                assert np.allclose(ds[f"welg_0_aux_l{layer + 1}"].values, FILL_DNODATA)
                assert ds[f"welg_0_q_l{layer + 1}"].dims == ("time", "nmesh_face")
                assert ds[f"welg_0_aux_l{layer + 1}"].dims == ("time", "nmesh_face")
            assert ds.sizes["time"] == 2
            assert ds.sizes["nmesh_face"] == 4
            assert len(ds) == 6


def test_om_param_mesh():
    # TODO: layer needed?
    params = [
        {
            "name": "aux",
            "attrs": {"layer": 1},
        },
        # {
        #    "name": "aux",
        #    "attrs": {"layer": 2},
        # },
        {
            "name": "q",
            "attrs": {"layer": 1},
        },
    ]
    context = {
        "mesh": "layered",
        "modelname": "gwfmodel",
        "grid": "structured",
        "package_name": "welg0",
        "package_type": "gwf-welg",
        "dims": [1, 1, 1, 1],
    }

    for p in params:
        pinst = NetCDFParam.from_meta(p, context=context)
        print(pinst.meta)
        ds = pinst.to_xarray()
        print(ds)
