/**
 * @file Operator.h
 * @brief Logical operator dispatch over recipes and executable providers.
 *
 * This layer separates inexpensive applicability checks from potentially
 * stateful artifact resolution.  It owns dispatch metadata but never tensor
 * storage.  All calls are synchronous; callers must synchronize aliased
 * writable tensor storage and implementations that do not document re-entry.
 */
#ifndef VECOPS_RUNTIME_OPERATOR_H
#define VECOPS_RUNTIME_OPERATOR_H

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "vecops/runtime/Executable.h"
#include "vecops/runtime/KernelDefinition.h"
#include "vecops/runtime/Schema.h"

namespace vecops::runtime {

/**
 * @brief Concrete recipe identity selected for one argument specialization.
 *
 * Both strings are copied values.  Providers use `recipe_id` to locate a
 * family and must return an executable whose descriptor key equals
 * `specialization_key`.
 */
struct BoundKernelRecipe {
  /** @brief Stable identifier of the recipe implementation family. */
  std::string recipe_id;
  /** @brief Exact specialization identity requested from a provider. */
  std::string specialization_key;
  /** @brief Stable recipe/compiler namespace component for persistent caches. */
  std::string artifact_key;
  /** @brief Optional compiler input for a source-backed recipe. */
  struct SourceInstantiation {
    /** @brief Absolute or recipe-relative source file defining `__kernel__`. */
    std::filesystem::path kernel_file;
    /** @brief Shared declarative interface used to generate this artifact. */
    std::shared_ptr<const KernelDef> definition;
    /** @brief Fully inferred metadata and specialization consumed by the compiler. */
    BoundKernel binding;
    /** @brief Logical task count compiled into `vecops::spec::Parallelism`. */
    nint_t parallelism = 1;
  };
  /** @brief Present only when a provider can compile this recipe from source. */
  std::shared_ptr<const SourceInstantiation> source;
};

/** Return a source recipe rebound to one compile-time logical task count. */
[[nodiscard]] BoundKernelRecipe specialize_parallelism(BoundKernelRecipe recipe, nint_t parallelism);

/**
 * @brief Stateless description of how one implementation family is specialized.
 *
 * Implementations must permit concurrent calls unless they document stricter
 * requirements.  `match` and `bind` receive non-owning argument metadata that
 * remains valid only for their call; they must not retain references or invoke
 * a build system.
 */
class KernelRecipe {
public:
  /** @brief Destroy through the abstract interface. */
  virtual ~KernelRecipe() = default;
  /**
   * @brief Return the stable recipe-family identifier.
   * @return View whose storage must outlive the recipe object.
   */
  [[nodiscard]] virtual std::string_view id() const = 0;
  /**
   * @brief Test cheap, local applicability to @p arguments.
   * @param arguments Non-owning ordered invocation metadata.
   * @return Success if `bind` may be attempted; otherwise `NotApplicable` or error.
   *
   * Must be fast, side-effect free, thread-safe for concurrent calls, and must
   * never invoke a build system.
   */
  [[nodiscard]] virtual Status match(const KernelCall& call) const = 0;
  /**
   * @brief Bind a matching argument sequence to an exact specialization.
   * @param arguments Non-owning ordered invocation metadata.
   * @return Recipe and specialization identifiers, or an error status.
   *
   * Must not retain argument references or mutate tensor storage.
   */
  [[nodiscard]] virtual Result<BoundKernelRecipe> bind(const KernelCall& call) const = 0;
};

/** @brief Recipe for one compiler-managed C++ file defining `__kernel__`. */
class SourceKernelRecipe final : public KernelRecipe {
public:
  /**
   * @brief Construct a source recipe and freeze its cache identity.
   *
   * When `source_fingerprint` is empty the constructor reads `kernel_file`
   * once and fingerprints its contents. Source-free cache-only deployments
   * should pass the fingerprint recorded when artifacts were produced.
   */
  SourceKernelRecipe(std::string id, std::filesystem::path kernel_file, std::shared_ptr<const KernelDef> definition,
                     std::string source_fingerprint = {}, SpecializationValues default_values = {});

  /** @copydoc KernelRecipe::id */
  [[nodiscard]] std::string_view id() const override {
    return id_;
  }
  /** @copydoc KernelRecipe::match */
  [[nodiscard]] Status match(const KernelCall& call) const override;
  /** @copydoc KernelRecipe::bind */
  [[nodiscard]] Result<BoundKernelRecipe> bind(const KernelCall& call) const override;
  /** @brief Return the shared immutable declarative interface. */
  [[nodiscard]] const KernelDef& definition() const {
    return *definition_;
  }
  /** @brief Return the compiler input path retained by this recipe. */
  [[nodiscard]] const std::filesystem::path& kernel_file() const {
    return kernel_file_;
  }

private:
  std::string id_;
  std::filesystem::path kernel_file_;
  std::shared_ptr<const KernelDef> definition_;
  std::string source_fingerprint_;
  SpecializationValues default_values_;
};

/**
 * @brief Select and order candidate recipes for an invocation.
 *
 * Returned indices address the supplied `recipes` span and must be in range.
 * The span and argument metadata are transient; implementations must not keep
 * them.  Complex operators can provide policy while keeping match/bind logic
 * in `KernelRecipe`.
 */
class DispatchPolicy {
public:
  /** @brief Destroy through the abstract interface. */
  virtual ~DispatchPolicy() = default;
  /**
   * @brief Produce an ordered candidate list.
   * @param arguments Non-owning invocation metadata.
   * @param recipes Available recipe objects indexed by the returned values.
   * @return Candidate indices, or a status when dispatch cannot proceed.
   *
   * Must not modify the span, recipes, arguments, or tensor storage.
   */
  [[nodiscard]] virtual Result<std::vector<std::size_t>>
  candidates(const KernelCall& call, std::span<const std::shared_ptr<const KernelRecipe>> recipes) const = 0;
};

/**
 * @brief Dispatch policy preserving the caller's recipe declaration order.
 *
 * Its `candidates` result includes every supplied recipe exactly once.  It is
 * stateless and safe for concurrent immutable use.
 */
class OrderedDispatchPolicy final : public DispatchPolicy {
public:
  /** @copydoc DispatchPolicy::candidates */
  [[nodiscard]] Result<std::vector<std::size_t>>
  candidates(const KernelCall& call, std::span<const std::shared_ptr<const KernelRecipe>> recipes) const override;
};

/**
 * @brief Resolve a bound recipe from user-selected loaded, AOT, or JIT providers.
 *
 * Resolution may load or compile artifacts and therefore may have filesystem,
 * process, and cache side effects.  Implementations own policy for caching and
 * concurrency, but must return an executable that independently retains its
 * backing module.
 */
class ExecutableProvider {
public:
  /** @brief Destroy through the abstract interface. */
  virtual ~ExecutableProvider() = default;
  /**
   * @brief Resolve or create the requested specialization.
   * @param recipe Bound recipe identity to satisfy.
   * @return Independently-owned executable or an error status.
   *
   * May perform I/O, dynamic loading, compilation, or cache mutation.  It must
   * not return an executable whose descriptor key differs from the request.
   */
  [[nodiscard]] virtual Result<std::shared_ptr<Executable>> resolve(const BoundKernelRecipe& recipe) = 0;
};

/**
 * @brief Stable logical operator combining schema, dispatch, and provider.
 *
 * The operator owns shared references to its recipes, dispatch policy, and
 * provider.  Generated `Executable` instances remain usable after this object
 * is destroyed.  Concurrent invocation safety depends on those collaborators
 * and the selected kernel; this class does not serialize them.
 */
class Operator {
public:
  /**
   * @brief Construct a complete logical operator.
   * @param schema Owning logical schema.
   * @param recipes Candidate implementation families.
   * @param dispatch_policy Policy used to order candidate recipes.
   * @param provider Resolver for bound specializations; must not be null.
   */
  Operator(OperatorSchema schema, std::vector<std::shared_ptr<const KernelRecipe>> recipes,
           std::shared_ptr<const DispatchPolicy> dispatch_policy, std::shared_ptr<ExecutableProvider> provider);

  /** @brief Construct an operator whose source recipes share one declarative definition. */
  Operator(KernelDef definition, std::vector<std::shared_ptr<const KernelRecipe>> recipes,
           std::shared_ptr<const DispatchPolicy> dispatch_policy, std::shared_ptr<ExecutableProvider> provider);

  /** @brief Return the owned immutable operator schema. */
  [[nodiscard]] const OperatorSchema& schema() const {
    return schema_;
  }
  /**
   * @brief Validate, dispatch, bind, and resolve an executable.
   * @param arguments Ordered metadata to inspect during dispatch.
   * @return Independently-owned executable or a validation/dispatch/provider error.
   *
   * Provider resolution may have the side effects documented by
   * `ExecutableProvider::resolve`; no tensor storage is accessed here.
   */
  [[nodiscard]] Result<std::shared_ptr<Executable>> resolve(const ArgumentMetadata& arguments) const;
  /**
   * @brief Validate, bind, dispatch, and resolve a call with explicit specialization values.
   * @param call Positional metadata plus values not inferred from tensor metadata.
   * @return Independently-owned executable or an explanatory failure status.
   */
  [[nodiscard]] Result<std::shared_ptr<Executable>> resolve(
    const KernelCall& call, const VecopsExecutionContext* context = nullptr) const;
  /**
   * @brief Resolve then synchronously execute an applicable specialization.
   * @param arguments Ordered metadata and externally-owned tensor storage.
   * @param workspace Caller-owned workspace, or null.
   * @param workspace_size Workspace capacity in bytes.
   * @param context Optional non-owning context valid during execution.
   * @return Success or validation, resolution, or execution failure.
   *
   * May trigger provider side effects and may write declared output tensors.
   */
  [[nodiscard]] Status invoke(const ArgumentMetadata& arguments, void* workspace = nullptr,
                              std::uint64_t workspace_size = 0, const VecopsExecutionContext* context = nullptr) const;
  /**
   * @brief Resolve and execute a KernelDef-aware call.
   * @param call Positional metadata plus explicit specialization values.
   * @param context Optional non-owning execution context valid during execution.
   * @param workspace Optional caller-owned external workspace allocation.
   * @param workspace_size Available external workspace bytes.
   *
   * The trailing workspace arguments preserve the historical
   * `invoke(call, context)` source API while allowing framework bridges to
   * forward the complete `VecopsCall` frame. Source kernels that own all
   * temporary allocation internally continue to use the default null/zero
   * workspace. The call is synchronous and may write `Output` and `InOut`
   * tensor arguments.
   */
  [[nodiscard]] Status invoke(const KernelCall& call, const VecopsExecutionContext* context = nullptr,
                              void* workspace = nullptr, std::uint64_t workspace_size = 0) const;

  /** @brief Shorthand for the allocation-free external-workspace kernel path. */
  [[nodiscard]] Status operator()(const KernelCall& call) const {
    return invoke(call);
  }

  /**
   * @brief Type-erase values, then call `invoke`.
   * @param values Tensors, scalars, tensor views, or omitted optional values.
   * @return Same status contract as `invoke` with null workspace and context.
   *
   * Tensor storage must remain alive through the synchronous call.
   */
  template <typename... Values>
  [[nodiscard]] Status invoke_values(Values&&... values) const {
    return invoke(make_arguments(std::forward<Values>(values)...));
  }

private:
  [[nodiscard]] Result<KernelCall> normalize(const KernelCall& call) const;
  /** Resolve a call that has already passed this operator's normalization. */
  [[nodiscard]] Result<std::shared_ptr<Executable>> resolve_normalized(
    const KernelCall& call, const VecopsExecutionContext* context) const;
  OperatorSchema schema_;
  std::shared_ptr<const KernelDef> kernel_definition_;
  std::vector<std::shared_ptr<const KernelRecipe>> recipes_;
  std::shared_ptr<const DispatchPolicy> dispatch_policy_;
  std::shared_ptr<ExecutableProvider> provider_;
};

} // namespace vecops::runtime

#endif // VECOPS_RUNTIME_OPERATOR_H
