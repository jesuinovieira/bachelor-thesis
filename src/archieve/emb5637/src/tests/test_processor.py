import pytest


@pytest.mark.unit
def test_transform_ok():
    # Use MinMaxScaler (or StandardScaler, or both) here, and assert that self.df is the
    # same after calling transform
    pass


@pytest.mark.unit
def test_fit_ok():
    # Assert the length of historical data (yhat and ytrue) in self.pr
    pass


@pytest.mark.unit
def test_predict_ok():
    # Create a ModelMocker that certifies predict() is called
    pass


@pytest.mark.unit
def test_add_ok():
    pass


@pytest.mark.unit
def test_add_error():
    pass


@pytest.mark.unit
def test_save_ok():
    # Parametrize: folder exist and folder don't exist
    # Clean up
    pass
