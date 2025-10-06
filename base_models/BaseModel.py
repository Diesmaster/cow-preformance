from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class for regression-like models.
    Defines the common public interface that all model subclasses (e.g., OLSModel, RidgeModel)
    must implement. Each method can be overridden with model-specific behavior.
    """

    def __init__(self, independent_attrs, dependent_attr, n, title):
        if isinstance(independent_attrs, str):
            self.independent_attrs = [independent_attrs]
        else:
            self.independent_attrs = independent_attrs

        self.dependent_attr = dependent_attr
        self.n = n
        self.title = title
        self.model = None
        self.results = None
        self.formula = None
        self.cv_results = None
        self.diagnostics = None

    # -------- abstract methods (interface definition) --------
    @abstractmethod
    def fit(self, df):
        """Fit the model to data."""
        pass

    @abstractmethod
    def predict(self, df):
        """Make predictions using the fitted model."""
        pass

    @abstractmethod
    def evaluate(self, df):
        """Evaluate model performance on a dataset."""
        pass

    @abstractmethod
    def cross_validate(self, df, k=5, random_state=42):
        """Perform cross-validation."""
        pass

    @abstractmethod
    def fit_with_cv(self, df, k=5, random_state=42):
        """Run cross-validation then fit full model."""
        pass

    @abstractmethod
    def summary(self):
        """Print or return a model summary."""
        pass

    @abstractmethod
    def print_diagnostics(self, show_arrays=False):
        """Print diagnostic test results."""
        pass

    @abstractmethod
    def save_results(self):
        """Save model results to JSON."""
        pass

    @abstractmethod
    def plot(self, df, save=True):
        """Plot actual vs predicted or regression line."""
        pass

    @abstractmethod
    def get_coefficients(self):
        """Return model coefficients."""
        pass

