import abc
from pathlib import Path

import numpy as np
import xarray as xr
from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator

from modflow_devtools.dfn import fetch, load

SPEC_PATH = None
FILL_DNODATA = np.float64(3e30)  # MF6 DNODATA constant
FILL_INT32 = np.int32(-2147483647)  # netcdf-fortran NF90_FILL_INT
FILL_INT64 = np.int64(-2147483647)  # netcdf-fortran NF90_FILL_INT
FILL_FLOAT64 = np.float64(9.96920996838687e36)  # netcdf-fortran NF90_FILL_DOUBLE


def get_dfn(toml_name):
    global SPEC_PATH
    if SPEC_PATH is None:
        SPEC_PATH = fetch.fetch_versioned_path()
    path = Path(SPEC_PATH / "toml" / f"{toml_name}.toml")
    if not path.is_file():
        raise AssertionError(f"Not a valid mf6 component: {toml_name}")
    with path.open(mode="rb") as toml_file:
        return load(toml_file, format="toml", name=toml_name)


# Define an Abstract Base Class (interface)
class NetCDFInput(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def from_meta(cls, meta: dict, context: dict | None):
        """Create new instance, validate against schema."""
        pass

    @abc.abstractmethod
    def to_xarray(self) -> xr.Dataset:
        """Xarray dataset."""
        pass

    @property
    @abc.abstractmethod
    def meta(self):
        """Abstract meta property getter."""
        pass


class NetCDFModel(BaseModel, NetCDFInput):
    attrs: "NetCDFModelAttrs"
    packages: list["NetCDFPackage"] = Field(default_factory=list)

    def model_post_init(self, __context) -> None:
        self._context = __context

    @classmethod
    def from_meta(cls, meta, context=None):
        try:
            if context:
                context = {
                    k.lower(): v.lower() if isinstance(v, str) else v
                    for k, v in context.items()
                }
            _meta = NetCDFModel._backfill_meta(meta, context)
            inst = cls.model_validate(_meta, context=context)
            inst._context |= context if context is not None else inst._context
            return inst
        except ValidationError:
            raise

    def to_xarray(self) -> xr.Dataset:
        dss = []
        meta = self.model_dump(by_alias=True)
        for p in self.packages:
            dss.append(p.to_xarray())

        ds = xr.merge(dss)
        for a in meta["attrs"]:
            if meta["attrs"][a] is not None:
                ds.attrs[a] = meta["attrs"][a]
        return ds

    @property
    def meta(self):
        """meta property getter."""
        return self.model_dump(by_alias=True)

    @field_validator("attrs", mode="before")
    @classmethod
    def validate_attrs(cls, v: dict[str, str]) -> dict[str, str]:
        """
        validate model (dataset) scoped attributes dictionary
        """
        v = {k.lower(): v.lower() if isinstance(v, str) else v for k, v in v.items()}
        return v

    @staticmethod
    def _backfill_meta(meta: dict, context: dict, verbose: bool = True) -> dict:
        _meta = dict(meta)

        if (
            "modflow_grid" not in _meta["attrs"]
            or "modflow_model" not in _meta["attrs"]
        ):
            raise AssertionError(
                "Model missing required modflow_grid or modflow_model attribute(s)."
            )
        mname = _meta["attrs"]["modflow_model"].split(":")[1].strip()

        _packages = []
        for pkg in _meta["packages"]:
            pkgctx = (
                {"mesh": _meta["attrs"]["mesh"]} if "mesh" in _meta["attrs"] else {}
            )
            pkgctx["modelname"] = mname
            pkgctx |= context
            _packages.append(NetCDFPackage.from_meta(pkg, context=pkgctx))
        _meta["packages"] = _packages

        return _meta


class NetCDFModelAttrs(BaseModel):
    # order of params dictates when data added to info dict
    mesh: str | None = Field(default=None)
    modflow_grid: str = Field()
    modflow_model: str = Field()

    @field_validator("mesh", mode="before")
    @classmethod
    def validate_mesh(cls, v: str | None, info: ValidationInfo) -> str | None:
        if v is not None:
            v = v.lower()
            assert v == "layered"
            info.context["mesh"] = v  # type: ignore
        return v

    @field_validator("modflow_grid", mode="before")
    @classmethod
    def validate_modflow_grid(cls, v: str, info: ValidationInfo) -> str:
        v = v.lower()
        mesh = info.data.get("mesh")
        dims = info.context.get("dims")  # type: ignore
        if mesh is None:
            assert v == "structured"
            if len(dims) != 4:
                raise AssertionError(f"Expected 4 input dimensions: {dims}")
        else:
            if len(dims) != 3:
                raise AssertionError(
                    f"Expected 3 input dimensions for layered mesh: {dims}"
                )
        info.context["grid"] = v  # type: ignore
        return v

    @field_validator("modflow_model", mode="before")
    @classmethod
    def validate_modflow_model(cls, v: str, info: ValidationInfo) -> str:
        v = v.lower()
        tokens = v.split(":")
        if len(tokens) != 2:
            raise ValueError(f"Invalid modflow_model attribute: {v}")
        modeltype = tokens[0].strip()
        if modeltype[-1].isdigit():
            modeltype = modeltype[:-1]
        info.context["modeltype"] = modeltype  # type: ignore
        info.context["modelname"] = tokens[1].strip()  # type: ignore
        return v


class NetCDFPackage(BaseModel, NetCDFInput):
    package_name: str = Field()
    package_type: str = Field()
    params: list["NetCDFParam"] = Field(default_factory=list)

    def model_post_init(self, __context) -> None:
        self._context = __context

    @classmethod
    def from_meta(cls, meta, context):
        try:
            if context:
                context = {
                    k.lower(): v.lower() if isinstance(v, str) else v
                    for k, v in context.items()
                }
            _meta = NetCDFPackage._backfill_meta(meta, context)
            inst = cls.model_validate(_meta, context=context)
            inst._context |= context if context is not None else inst._context
            return inst
        except ValidationError:
            raise

    def to_xarray(self) -> xr.Dataset:
        ds = []
        for p in self.params:
            ds.append(p.to_xarray())

        return xr.merge(ds)

    @property
    def meta(self):
        """meta property getter."""
        return self.model_dump(by_alias=True)

    @field_validator("package_type", mode="before")
    @classmethod
    def validate_ptype(cls, v: str, info: ValidationInfo) -> str:
        """
        validate package parameter identifier
        """
        v = v.lower()
        assert get_dfn(v)
        info.context["package_type"] = v  # type: ignore
        return v

    @staticmethod
    def _backfill_meta(meta: dict, context: dict, verbose: bool = True) -> dict:
        _meta = dict(meta)

        if "package_name" not in _meta or "package_type" not in _meta:
            raise AssertionError(
                "Package missing required package_name or package_type attribute(s)."
            )

        paramctx = dict(context)
        paramctx["package_name"] = _meta["package_name"]
        paramctx["package_type"] = _meta["package_type"]

        dims = context.get("dims", None)
        mesh = context.get("mesh", None)

        assert dims is not None

        # TODO auxiliary in context
        _params = []
        for p in _meta["params"]:
            if mesh is None:
                if "attrs" in p and "layer" in p["attrs"]:
                    assert p["attrs"]["layer"] is None
                _params.append(NetCDFParam.from_meta(p, context=paramctx))
            else:
                if "attrs" not in p:
                    p["attrs"] = {}
                for layer in range(dims[1]):
                    p["attrs"]["layer"] = layer + 1
                    _params.append(NetCDFParam.from_meta(p, context=paramctx))

        _meta["params"] = _params

        return _meta


class NetCDFParam(BaseModel, NetCDFInput):
    name: str = Field()
    shape: list[str] = Field(default_factory=list)
    attrs: "NetCDFParamAttrs"
    encodings: "NetCDFParamEncodings"
    dtype: str = Field()

    def model_post_init(self, __context) -> None:
        self._context = __context

    @classmethod
    def from_meta(cls, meta, context):
        try:
            if context:
                context = {
                    k.lower(): v.lower() if isinstance(v, str) else v
                    for k, v in context.items()
                }
            _meta = NetCDFParam._backfill_meta(meta, context)
            inst = cls.model_validate(_meta, context=context)
            inst._context |= context if context is not None else inst._context
            return inst
        except ValidationError:
            raise

    def to_xarray(self) -> xr.Dataset:
        dimmap = {
            "time": 0,
            "z": 1,
            "y": 2,
            "nmesh_face": 2,
            "x": 3,
        }

        dtype: type[np.generic]
        meta = self.model_dump(by_alias=True)
        ds = xr.Dataset()

        pname = self._context["package_name"]
        package_type = self._context["package_type"]
        mesh = self._context.get("mesh", None)
        ptype = package_type.split("-")[1].strip()
        dfn = get_dfn(package_type)
        param = (
            meta["name"]
            if mesh is None
            else f"{meta['name']}_l{meta['attrs']['layer']}"
        )
        if dfn.multi:
            varname = f"{pname}_{param}"
        else:
            varname = f"{ptype}_{param}"
        if meta["dtype"] == "float64":
            dtype = np.float64
        elif meta["dtype"] == "int64":
            dtype = np.int64
        dims = [self._context["dims"][dimmap[dim]] for dim in meta["shape"]]
        data = np.full(
            dims,
            meta["encodings"]["_FillValue"],
            dtype=dtype,
        )
        var_d = {varname: (meta["shape"], data)}
        ds = ds.assign(var_d)
        for a in meta["attrs"]:
            if meta["attrs"][a] is not None:
                ds[varname].attrs[a] = meta["attrs"][a]
        for e in meta["encodings"]:
            if meta["encodings"][e] is not None:
                ds[varname].encoding[e] = meta["encodings"][e]

        return ds

    @property
    def meta(self):
        """meta property getter."""
        return self.model_dump(by_alias=True)

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, v: str, info: ValidationInfo) -> str:
        """
        validate package parameter identifier
        """
        v = v.lower()
        package = info.context.get("package_type")  # type: ignore
        dfn = get_dfn(package)

        blocks = ["griddata", "period"]
        if not any(blk in dfn.blocks for blk in blocks):
            raise ValueError(
                f"griddata/period blocks not found in package type {package}"
            )
        if not any(
            v in dfn.blocks[blk] if blk in dfn.blocks else False for blk in blocks
        ):
            raise ValueError(f"Param {v} not found in package {package}")

        for b in blocks:
            if b in dfn.blocks and v in dfn.blocks[b]:
                if not dfn.blocks[b][v].netcdf:
                    raise ValueError(f"Not a netcdf param: '{v}'")
        return v

    @field_validator("shape", mode="before")
    @classmethod
    def validate_shape(cls, v: list[str]) -> list[str]:
        """
        validate parameter shape
        """
        v = [dim.lower() if isinstance(dim, str) else dim for dim in v]
        valid = ["time", "nmesh_face", "z", "y", "x"]
        if not all(dim in valid for dim in v):
            raise AssertionError(f"Invalid param shape={v}. Valid dims={valid}.")
        return v

    @field_validator("attrs", mode="before")
    @classmethod
    def validate_attrs(
        cls, v: dict[str, int | str], info: ValidationInfo
    ) -> dict[str, int | str]:
        """
        validate parameter attributes dictionary
        """
        v = {k.lower(): v.lower() if isinstance(v, str) else v for k, v in v.items()}
        param = info.data.get("name")
        # shape = info.data.get("shape")
        mesh = info.context.get("mesh")  # type: ignore
        if mesh is not None and ("layer" not in v or v["layer"] is None):
            raise AssertionError(f"Expected layer attribute for mesh param '{param}'")
        if param is not None and param == "aux" and "modflow_iaux" not in v:
            # TODO
            pass
            # raise AssertionError(
            #    f"Expected modflow_iaux attribute for aux param '{param}'"
            # )
        return v

    @field_validator("dtype", mode="before")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        """
        validate parameter shape
        """
        v = v.lower()
        valid = ["float64", "int64", "int32"]
        if v not in valid:
            raise AssertionError(f"Invalid param dtype={v}. Valid types={valid}.")
        return v

    @staticmethod
    def _backfill_meta(meta: dict, context: dict, verbose: bool = True) -> dict:
        _meta = dict(meta)

        def _structured_shape(dfn_shape):
            shape = ["time"] if "nper" in dfn_shape else []
            if "nodes" in dfn_shape:
                shape += ["z", "y", "x"]
            elif "ncpl" in dfn_shape:
                shape += ["y", "x"]
            else:
                if "nlay" in dfn_shape:
                    shape.append("z")
                if "nrow" in dfn_shape:
                    shape.append("y")
                if "ncol" in dfn_shape:
                    shape.append("x")
            return shape

        def _mesh_shape(dfn_shape):
            shape = ["time"] if "nper" in dfn_shape else []
            if (
                "nodes" in dfn_shape
                or "ncpl" in dfn_shape
                or ("nrow" in dfn_shape and "ncol" in dfn_shape)
            ):
                shape.append("nmesh_face")
            elif "nrow" in dfn_shape:
                shape.append("y")
            elif "ncol" in dfn_shape:
                shape.append("x")
            return shape

        mname = context["modelname"]
        mesh = context.get("mesh", None)
        dfn = get_dfn(context["package_type"])
        param = _meta["name"]
        if param not in dfn.fields:
            raise ValueError(
                f"Param {param} not found in package {context['package_type']}"
            )
        if "attrs" not in _meta:
            _meta["attrs"] = {}
        if "encodings" not in _meta:
            _meta["encodings"] = {}
        if "dtype" not in _meta:
            if dfn.fields[param].type == "double":
                _meta["dtype"] = "float64"
                if "_FillValue" not in _meta["encodings"]:
                    if dfn.fields[param].block == "period":
                        _meta["encodings"]["_FillValue"] = FILL_DNODATA
                    else:
                        _meta["encodings"]["_FillValue"] = FILL_FLOAT64
            elif dfn.fields[param].type == "integer":
                _meta["dtype"] = "int64"
                if "_FillValue" not in _meta["encodings"]:
                    if dfn.fields[param].block == "period":
                        _meta["encodings"]["_FillValue"] = FILL_DNODATA
                    else:
                        _meta["encodings"]["_FillValue"] = FILL_INT64
        if "modflow_input" not in _meta["attrs"]:
            if dfn.multi:
                _meta["attrs"]["modflow_input"] = (
                    f"{mname}/{context['package_name']}/{param}"
                )
            else:
                _meta["attrs"]["modflow_input"] = (
                    f"{mname}/{context['package_type']}/{param}"
                )
        if "shape" not in _meta:
            if mesh is not None:
                _meta["shape"] = _mesh_shape(dfn.fields[param].shape)
            else:
                _meta["shape"] = _structured_shape(dfn.fields[param].shape)

        return _meta


class NetCDFParamAttrs(BaseModel):
    modflow_input: str = Field()
    modflow_iaux: int | None = Field(default=None)
    layer: int | None = Field(default=None)

    @field_validator("modflow_input", mode="before")
    @classmethod
    def validate_modflow_input(cls, v: str, info: ValidationInfo) -> str:
        v = v.lower()
        modelname = info.context.get("modelname")  # type: ignore
        if v.split("/")[0] != modelname:
            raise ValueError(
                f'modflow_input attribute "{v}" does not '
                f'match dataset modelname "{modelname}")'
            )
        return v

    @field_validator("modflow_iaux", mode="before")
    @classmethod
    def validate_modflow_iaux(cls, v: int, info: ValidationInfo) -> int:
        return v

    @field_validator("layer", mode="before")
    @classmethod
    def validate_layer(cls, v: int, info: ValidationInfo) -> int:
        dims = info.context.get("dims")  # type: ignore
        if v is not None and v > dims[1]:
            raise ValueError(f"Param layer attribute value {v} exceeds grid k")
        return v


class NetCDFParamEncodings(BaseModel):
    fill: float = Field(alias="_FillValue")

    @field_validator("fill", mode="before")
    @classmethod
    def validate_fill(cls, v: float, info: ValidationInfo) -> float:
        return v
