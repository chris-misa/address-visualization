{
  lib,
  buildPythonPackage,
  fetchPypi,
  setuptools,
  wheel,
}:

buildPythonPackage rec {
  pname = "hilbertcurve";
  version = "2.0.5";

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-ancD2aLx/nSMhthpCL8YPn0Tm5c2ReSyUm4Qs051eW0=";
  };

  # do not run tests
  doCheck = false;

  # specific to buildPythonPackage, see its reference
  pyproject = true;
  build-system = [
    setuptools
    wheel
  ];

  dontCheckRuntimeDeps = true;
}