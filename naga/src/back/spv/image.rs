/*!
Generating SPIR-V for image operations.
*/

use spirv::Word;

use super::{
    instructions::SampleLod,
    selection::{MergeTuple, Selection},
    Block, BlockContext, Error, IdGenerator, Instruction, LocalType, LookupType, NumericType,
    WrappedFunction,
};
use crate::{arena::Handle, common::ExternalTextureParameter};

/// Information about a vector of coordinates.
///
/// The coordinate vectors expected by SPIR-V `OpImageRead` and `OpImageFetch`
/// supply the array index for arrayed images as an additional component at
/// the end, whereas Naga's `ImageLoad`, `ImageStore`, and `ImageSample` carry
/// the array index as a separate field.
///
/// In the process of generating code to compute the combined vector, we also
/// produce SPIR-V types and vector lengths that are useful elsewhere. This
/// struct gathers that information into one place, with standard names.
struct ImageCoordinates {
    /// The SPIR-V id of the combined coordinate/index vector value.
    ///
    /// Note: when indexing a non-arrayed 1D image, this will be a scalar.
    value_id: Word,

    /// The SPIR-V id of the type of `value`.
    type_id: Word,

    /// The number of components in `value`, if it is a vector, or `None` if it
    /// is a scalar.
    size: Option<crate::VectorSize>,
}

/// A trait for image access (load or store) code generators.
///
/// Types implementing this trait hold information about an `ImageStore` or
/// `ImageLoad` operation that is not affected by the bounds check policy. The
/// `generate` method emits code for the access, given the results of bounds
/// checking.
///
/// The [`image`] bounds checks policy affects access coordinates, level of
/// detail, and sample index, but never the image id, result type (if any), or
/// the specific SPIR-V instruction used. Types that implement this trait gather
/// together the latter category, so we don't have to plumb them through the
/// bounds-checking code.
///
/// [`image`]: crate::proc::BoundsCheckPolicies::index
trait Access {
    /// The Rust type that represents SPIR-V values and types for this access.
    ///
    /// For operations like loads, this is `Word`. For operations like stores,
    /// this is `()`.
    ///
    /// For `ReadZeroSkipWrite`, this will be the type of the selection
    /// construct that performs the bounds checks, so it must implement
    /// `MergeTuple`.
    type Output: MergeTuple + Copy + Clone;

    /// Write an image access to `block`.
    ///
    /// Access the texel at `coordinates_id`. The optional `level_id` indicates
    /// the level of detail, and `sample_id` is the index of the sample to
    /// access in a multisampled texel.
    ///
    /// This method assumes that `coordinates_id` has already had the image array
    /// index, if any, folded in, as done by `write_image_coordinates`.
    ///
    /// Return the value id produced by the instruction, if any.
    ///
    /// Use `id_gen` to generate SPIR-V ids as necessary.
    fn generate(
        &self,
        id_gen: &mut IdGenerator,
        coordinates_id: Word,
        level_id: Option<Word>,
        sample_id: Option<Word>,
        block: &mut Block,
    ) -> Self::Output;

    /// Return the SPIR-V type of the value produced by the code written by
    /// `generate`. If the access does not produce a value, `Self::Output`
    /// should be `()`.
    fn result_type(&self) -> Self::Output;

    /// Construct the SPIR-V 'zero' value to be returned for an out-of-bounds
    /// access under the `ReadZeroSkipWrite` policy. If the access does not
    /// produce a value, `Self::Output` should be `()`.
    fn out_of_bounds_value(&self, ctx: &mut BlockContext<'_>) -> Self::Output;
}

/// Texel access information for an [`ImageLoad`] expression.
///
/// [`ImageLoad`]: crate::Expression::ImageLoad
struct Load {
    /// The specific opcode we'll use to perform the fetch. Storage images
    /// require `OpImageRead`, while sampled images require `OpImageFetch`.
    opcode: spirv::Op,

    /// The type id produced by the actual image access instruction.
    type_id: Word,

    /// The id of the image being accessed.
    image_id: Word,
}

impl Load {
    fn from_image_expr(
        ctx: &mut BlockContext<'_>,
        image_id: Word,
        image_class: crate::ImageClass,
        result_type_id: Word,
    ) -> Result<Load, Error> {
        let opcode = match image_class {
            crate::ImageClass::Storage { .. } => spirv::Op::ImageRead,
            crate::ImageClass::Depth { .. } | crate::ImageClass::Sampled { .. } => {
                spirv::Op::ImageFetch
            }
            crate::ImageClass::External => unimplemented!(),
        };

        // `OpImageRead` and `OpImageFetch` instructions produce vec4<f32>
        // values. Most of the time, we can just use `result_type_id` for
        // this. The exception is that `Expression::ImageLoad` from a depth
        // image produces a scalar `f32`, so in that case we need to find
        // the right SPIR-V type for the access instruction here.
        let type_id = match image_class {
            crate::ImageClass::Depth { .. } => ctx.get_numeric_type_id(NumericType::Vector {
                size: crate::VectorSize::Quad,
                scalar: crate::Scalar::F32,
            }),
            _ => result_type_id,
        };

        Ok(Load {
            opcode,
            type_id,
            image_id,
        })
    }
}

impl Access for Load {
    type Output = Word;

    /// Write an instruction to access a given texel of this image.
    fn generate(
        &self,
        id_gen: &mut IdGenerator,
        coordinates_id: Word,
        level_id: Option<Word>,
        sample_id: Option<Word>,
        block: &mut Block,
    ) -> Word {
        let texel_id = id_gen.next();
        let mut instruction = Instruction::image_fetch_or_read(
            self.opcode,
            self.type_id,
            texel_id,
            self.image_id,
            coordinates_id,
        );

        match (level_id, sample_id) {
            (None, None) => {}
            (Some(level_id), None) => {
                instruction.add_operand(spirv::ImageOperands::LOD.bits());
                instruction.add_operand(level_id);
            }
            (None, Some(sample_id)) => {
                instruction.add_operand(spirv::ImageOperands::SAMPLE.bits());
                instruction.add_operand(sample_id);
            }
            // There's no such thing as a multi-sampled mipmap.
            (Some(_), Some(_)) => unreachable!(),
        }

        block.body.push(instruction);

        texel_id
    }

    fn result_type(&self) -> Word {
        self.type_id
    }

    fn out_of_bounds_value(&self, ctx: &mut BlockContext<'_>) -> Word {
        ctx.writer.get_constant_null(self.type_id)
    }
}

/// Texel access information for a [`Store`] statement.
///
/// [`Store`]: crate::Statement::Store
struct Store {
    /// The id of the image being written to.
    image_id: Word,

    /// The value we're going to write to the texel.
    value_id: Word,
}

impl Access for Store {
    /// Stores don't generate any value.
    type Output = ();

    fn generate(
        &self,
        _id_gen: &mut IdGenerator,
        coordinates_id: Word,
        _level_id: Option<Word>,
        _sample_id: Option<Word>,
        block: &mut Block,
    ) {
        block.body.push(Instruction::image_write(
            self.image_id,
            coordinates_id,
            self.value_id,
        ));
    }

    /// Stores don't generate any value, so this just returns `()`.
    fn result_type(&self) {}

    /// Stores don't generate any value, so this just returns `()`.
    fn out_of_bounds_value(&self, _ctx: &mut BlockContext<'_>) {}
}

impl BlockContext<'_> {
    /// Extend image coordinates with an array index, if necessary.
    ///
    /// Whereas [`Expression::ImageLoad`] and [`ImageSample`] treat the array
    /// index as a separate operand from the coordinates, SPIR-V image access
    /// instructions include the array index in the `coordinates` operand. This
    /// function builds a SPIR-V coordinate vector from a Naga coordinate vector
    /// and array index, if one is supplied, and returns a `ImageCoordinates`
    /// struct describing what it built.
    ///
    /// If `array_index` is `Some(expr)`, then this function constructs a new
    /// vector that is `coordinates` with `array_index` concatenated onto the
    /// end: a `vec2` becomes a `vec3`, a scalar becomes a `vec2`, and so on.
    ///
    /// If `array_index` is `None`, then the return value uses `coordinates`
    /// unchanged. Note that, when indexing a non-arrayed 1D image, this will be
    /// a scalar value.
    ///
    /// If needed, this function generates code to convert the array index,
    /// always an integer scalar, to match the component type of `coordinates`.
    /// Naga's `ImageLoad` and SPIR-V's `OpImageRead`, `OpImageFetch`, and
    /// `OpImageWrite` all use integer coordinates, while Naga's `ImageSample`
    /// and SPIR-V's `OpImageSample...` instructions all take floating-point
    /// coordinate vectors.
    ///
    /// [`Expression::ImageLoad`]: crate::Expression::ImageLoad
    /// [`ImageSample`]: crate::Expression::ImageSample
    fn write_image_coordinates(
        &mut self,
        coordinates: Handle<crate::Expression>,
        array_index: Option<Handle<crate::Expression>>,
        block: &mut Block,
    ) -> Result<ImageCoordinates, Error> {
        use crate::TypeInner as Ti;
        use crate::VectorSize as Vs;

        let coordinates_id = self.cached[coordinates];
        let ty = &self.fun_info[coordinates].ty;
        let inner_ty = ty.inner_with(&self.ir_module.types);

        // If there's no array index, the image coordinates are exactly the
        // `coordinate` field of the `Expression::ImageLoad`. No work is needed.
        let array_index = match array_index {
            None => {
                let value_id = coordinates_id;
                let type_id = self.get_expression_type_id(ty);
                let size = match *inner_ty {
                    Ti::Scalar { .. } => None,
                    Ti::Vector { size, .. } => Some(size),
                    _ => return Err(Error::Validation("coordinate type")),
                };
                return Ok(ImageCoordinates {
                    value_id,
                    type_id,
                    size,
                });
            }
            Some(ix) => ix,
        };

        // Find the component type of `coordinates`, and figure out the size the
        // combined coordinate vector will have.
        let (component_scalar, size) = match *inner_ty {
            Ti::Scalar(scalar @ crate::Scalar { width: 4, .. }) => (scalar, Vs::Bi),
            Ti::Vector {
                scalar: scalar @ crate::Scalar { width: 4, .. },
                size: Vs::Bi,
            } => (scalar, Vs::Tri),
            Ti::Vector {
                scalar: scalar @ crate::Scalar { width: 4, .. },
                size: Vs::Tri,
            } => (scalar, Vs::Quad),
            Ti::Vector { size: Vs::Quad, .. } => {
                return Err(Error::Validation("extending vec4 coordinate"));
            }
            ref other => {
                log::error!("wrong coordinate type {other:?}");
                return Err(Error::Validation("coordinate type"));
            }
        };

        // Convert the index to the coordinate component type, if necessary.
        let array_index_id = self.cached[array_index];
        let ty = &self.fun_info[array_index].ty;
        let inner_ty = ty.inner_with(&self.ir_module.types);
        let array_index_scalar = match *inner_ty {
            Ti::Scalar(
                scalar @ crate::Scalar {
                    kind: crate::ScalarKind::Sint | crate::ScalarKind::Uint,
                    width: 4,
                },
            ) => scalar,
            _ => unreachable!("we only allow i32 and u32"),
        };
        let cast = match (component_scalar.kind, array_index_scalar.kind) {
            (crate::ScalarKind::Sint, crate::ScalarKind::Sint)
            | (crate::ScalarKind::Uint, crate::ScalarKind::Uint) => None,
            (crate::ScalarKind::Sint, crate::ScalarKind::Uint)
            | (crate::ScalarKind::Uint, crate::ScalarKind::Sint) => Some(spirv::Op::Bitcast),
            (crate::ScalarKind::Float, crate::ScalarKind::Sint) => Some(spirv::Op::ConvertSToF),
            (crate::ScalarKind::Float, crate::ScalarKind::Uint) => Some(spirv::Op::ConvertUToF),
            (crate::ScalarKind::Bool, _) => unreachable!("we don't allow bool for component"),
            (_, crate::ScalarKind::Bool | crate::ScalarKind::Float) => {
                unreachable!("we don't allow bool or float for array index")
            }
            (crate::ScalarKind::AbstractInt | crate::ScalarKind::AbstractFloat, _)
            | (_, crate::ScalarKind::AbstractInt | crate::ScalarKind::AbstractFloat) => {
                unreachable!("abstract types should never reach backends")
            }
        };
        let reconciled_array_index_id = if let Some(cast) = cast {
            let component_ty_id = self.get_numeric_type_id(NumericType::Scalar(component_scalar));
            let reconciled_id = self.gen_id();
            block.body.push(Instruction::unary(
                cast,
                component_ty_id,
                reconciled_id,
                array_index_id,
            ));
            reconciled_id
        } else {
            array_index_id
        };

        // Find the SPIR-V type for the combined coordinates/index vector.
        let type_id = self.get_numeric_type_id(NumericType::Vector {
            size,
            scalar: component_scalar,
        });

        // Schmear the coordinates and index together.
        let value_id = self.gen_id();
        block.body.push(Instruction::composite_construct(
            type_id,
            value_id,
            &[coordinates_id, reconciled_array_index_id],
        ));
        Ok(ImageCoordinates {
            value_id,
            type_id,
            size: Some(size),
        })
    }

    pub(super) fn get_handle_id(&mut self, expr_handle: Handle<crate::Expression>) -> Word {
        let id = match self.ir_function.expressions[expr_handle] {
            crate::Expression::GlobalVariable(handle) => {
                self.writer.global_variables[handle].handle_id
            }
            crate::Expression::FunctionArgument(i) => {
                let offset = self.function.parameter_offsets[i as usize] as usize;
                self.function.parameters[offset].handle_id
            }
            crate::Expression::Access { .. } | crate::Expression::AccessIndex { .. } => {
                self.cached[expr_handle]
            }
            ref other => unreachable!("Unexpected image expression {:?}", other),
        };

        if id == 0 {
            unreachable!(
                "Image expression {:?} doesn't have a handle ID",
                expr_handle
            );
        }

        id
    }

    pub(super) fn get_external_texture_plane_id(&mut self, expr_handle: Handle<crate::Expression>, plane: usize) -> Word {
        assert!(plane < 3);
        let id = match self.ir_function.expressions[expr_handle] {
            crate::Expression::GlobalVariable(handle) => {
                self.writer.external_textures[&handle].planes[plane].var_id
            }
            crate::Expression::FunctionArgument(i) => {
                let offset = self.function.parameter_offsets[i as usize] as usize;
                self.function.parameters[offset + plane].handle_id
            }
            crate::Expression::Access { .. } | crate::Expression::AccessIndex { .. } => {
                // https://github.com/gfx-rs/wgpu/issues/8027
                unreachable!("External texture binding array")
            }
            ref other => unreachable!("Unexpected image expression {:?}", other),
        };

        if id == 0 {
            unreachable!(
                "External texture image expression {:?} doesn't have a handle ID for plane {plane}",
                expr_handle
            );
        }

        id
    }

    pub(super) fn get_external_texture_params_id(&mut self, expr_handle: Handle<crate::Expression>) -> Word {
        let id = match self.ir_function.expressions[expr_handle] {
            crate::Expression::GlobalVariable(handle) => {
                self.writer.external_textures[&handle].params.handle_id
            }
            crate::Expression::FunctionArgument(i) => {
                let offset = self.function.parameter_offsets[i as usize] as usize;
                self.function.parameters[offset + 3].handle_id
            }
            crate::Expression::Access { .. } | crate::Expression::AccessIndex { .. } => {
                // https://github.com/gfx-rs/wgpu/issues/8027
                unreachable!("External texture binding array")
            }
            ref other => unreachable!("Unexpected image expression {:?}", other),
        };

        if id == 0 {
            unreachable!(
                "External texture image expression {:?} doesn't have an access/handle ID for params",
                expr_handle
            );
        }

        id
    }

    /// Generate a vector or scalar 'one' for arithmetic on `coordinates`.
    ///
    /// If `coordinates` is a scalar, return a scalar one. Otherwise, return
    /// a vector of ones.
    fn write_coordinate_one(&mut self, coordinates: &ImageCoordinates) -> Result<Word, Error> {
        let one = self.get_scope_constant(1);
        match coordinates.size {
            None => Ok(one),
            Some(vector_size) => {
                let ones = [one; 4];
                let id = self.gen_id();
                Instruction::constant_composite(
                    coordinates.type_id,
                    id,
                    &ones[..vector_size as usize],
                )
                .to_words(&mut self.writer.logical_layout.declarations);
                Ok(id)
            }
        }
    }

    /// Generate code to restrict `input` to fall between zero and one less than
    /// `size_id`.
    ///
    /// Both must be 32-bit scalar integer values, whose type is given by
    /// `type_id`. The computed value is also of type `type_id`.
    fn restrict_scalar(
        &mut self,
        type_id: Word,
        input_id: Word,
        size_id: Word,
        block: &mut Block,
    ) -> Result<Word, Error> {
        let i32_one_id = self.get_scope_constant(1);

        // Subtract one from `size` to get the largest valid value.
        let limit_id = self.gen_id();
        block.body.push(Instruction::binary(
            spirv::Op::ISub,
            type_id,
            limit_id,
            size_id,
            i32_one_id,
        ));

        // Use an unsigned minimum, to handle both positive out-of-range values
        // and negative values in a single instruction: negative values of
        // `input_id` get treated as very large positive values.
        let restricted_id = self.gen_id();
        block.body.push(Instruction::ext_inst_gl_op(
            self.writer.gl450_ext_inst_id,
            spirv::GlslStd450Op::UMin,
            type_id,
            restricted_id,
            &[input_id, limit_id],
        ));

        Ok(restricted_id)
    }

    /// Write instructions to query the size of an image.
    ///
    /// This takes care of selecting the right instruction depending on whether
    /// a level of detail parameter is present.
    fn write_coordinate_bounds(
        &mut self,
        type_id: Word,
        image_id: Word,
        level_id: Option<Word>,
        block: &mut Block,
    ) -> Word {
        let coordinate_bounds_id = self.gen_id();
        match level_id {
            Some(level_id) => {
                // A level of detail was provided, so fetch the image size for
                // that level.
                let mut inst = Instruction::image_query(
                    spirv::Op::ImageQuerySizeLod,
                    type_id,
                    coordinate_bounds_id,
                    image_id,
                );
                inst.add_operand(level_id);
                block.body.push(inst);
            }
            _ => {
                // No level of detail was given.
                block.body.push(Instruction::image_query(
                    spirv::Op::ImageQuerySize,
                    type_id,
                    coordinate_bounds_id,
                    image_id,
                ));
            }
        }

        coordinate_bounds_id
    }

    /// Write code to restrict coordinates for an image reference.
    ///
    /// First, clamp the level of detail or sample index to fall within bounds.
    /// Then, obtain the image size, possibly using the clamped level of detail.
    /// Finally, use an unsigned minimum instruction to force all coordinates
    /// into range.
    ///
    /// Return a triple `(COORDS, LEVEL, SAMPLE)`, where `COORDS` is a coordinate
    /// vector (including the array index, if any), `LEVEL` is an optional level
    /// of detail, and `SAMPLE` is an optional sample index, all guaranteed to
    /// be in-bounds for `image_id`.
    ///
    /// The result is usually a vector, but it is a scalar when indexing
    /// non-arrayed 1D images.
    fn write_restricted_coordinates(
        &mut self,
        image_id: Word,
        coordinates: ImageCoordinates,
        level_id: Option<Word>,
        sample_id: Option<Word>,
        block: &mut Block,
    ) -> Result<(Word, Option<Word>, Option<Word>), Error> {
        self.writer.require_any(
            "the `Restrict` image bounds check policy",
            &[spirv::Capability::ImageQuery],
        )?;

        let i32_type_id = self.get_numeric_type_id(NumericType::Scalar(crate::Scalar::I32));

        // If `level` is `Some`, clamp it to fall within bounds. This must
        // happen first, because we'll use it to query the image size for
        // clamping the actual coordinates.
        let level_id = level_id
            .map(|level_id| {
                // Find the number of mipmap levels in this image.
                let num_levels_id = self.gen_id();
                block.body.push(Instruction::image_query(
                    spirv::Op::ImageQueryLevels,
                    i32_type_id,
                    num_levels_id,
                    image_id,
                ));

                self.restrict_scalar(i32_type_id, level_id, num_levels_id, block)
            })
            .transpose()?;

        // If `sample_id` is `Some`, clamp it to fall within bounds.
        let sample_id = sample_id
            .map(|sample_id| {
                // Find the number of samples per texel.
                let num_samples_id = self.gen_id();
                block.body.push(Instruction::image_query(
                    spirv::Op::ImageQuerySamples,
                    i32_type_id,
                    num_samples_id,
                    image_id,
                ));

                self.restrict_scalar(i32_type_id, sample_id, num_samples_id, block)
            })
            .transpose()?;

        // Obtain the image bounds, including the array element count.
        let coordinate_bounds_id =
            self.write_coordinate_bounds(coordinates.type_id, image_id, level_id, block);

        // Compute maximum valid values from the bounds.
        let ones = self.write_coordinate_one(&coordinates)?;
        let coordinate_limit_id = self.gen_id();
        block.body.push(Instruction::binary(
            spirv::Op::ISub,
            coordinates.type_id,
            coordinate_limit_id,
            coordinate_bounds_id,
            ones,
        ));

        // Restrict the coordinates to fall within those bounds.
        //
        // Use an unsigned minimum, to handle both positive out-of-range values
        // and negative values in a single instruction: negative values of
        // `coordinates` get treated as very large positive values.
        let restricted_coordinates_id = self.gen_id();
        block.body.push(Instruction::ext_inst_gl_op(
            self.writer.gl450_ext_inst_id,
            spirv::GlslStd450Op::UMin,
            coordinates.type_id,
            restricted_coordinates_id,
            &[coordinates.value_id, coordinate_limit_id],
        ));

        Ok((restricted_coordinates_id, level_id, sample_id))
    }

    fn write_conditional_image_access<A: Access>(
        &mut self,
        image_id: Word,
        coordinates: ImageCoordinates,
        level_id: Option<Word>,
        sample_id: Option<Word>,
        block: &mut Block,
        access: &A,
    ) -> Result<A::Output, Error> {
        self.writer.require_any(
            "the `ReadZeroSkipWrite` image bounds check policy",
            &[spirv::Capability::ImageQuery],
        )?;

        let bool_type_id = self.writer.get_bool_type_id();
        let i32_type_id = self.get_numeric_type_id(NumericType::Scalar(crate::Scalar::I32));

        let null_id = access.out_of_bounds_value(self);

        let mut selection = Selection::start(block, access.result_type());

        // If `level_id` is `Some`, check whether it is within bounds. This must
        // happen first, because we'll be supplying this as an argument when we
        // query the image size.
        if let Some(level_id) = level_id {
            // Find the number of mipmap levels in this image.
            let num_levels_id = self.gen_id();
            selection.block().body.push(Instruction::image_query(
                spirv::Op::ImageQueryLevels,
                i32_type_id,
                num_levels_id,
                image_id,
            ));

            let lod_cond_id = self.gen_id();
            selection.block().body.push(Instruction::binary(
                spirv::Op::ULessThan,
                bool_type_id,
                lod_cond_id,
                level_id,
                num_levels_id,
            ));

            selection.if_true(self, lod_cond_id, null_id);
        }

        // If `sample_id` is `Some`, check whether it is in bounds.
        if let Some(sample_id) = sample_id {
            // Find the number of samples per texel.
            let num_samples_id = self.gen_id();
            selection.block().body.push(Instruction::image_query(
                spirv::Op::ImageQuerySamples,
                i32_type_id,
                num_samples_id,
                image_id,
            ));

            let samples_cond_id = self.gen_id();
            selection.block().body.push(Instruction::binary(
                spirv::Op::ULessThan,
                bool_type_id,
                samples_cond_id,
                sample_id,
                num_samples_id,
            ));

            selection.if_true(self, samples_cond_id, null_id);
        }

        // Obtain the image bounds, including any array element count.
        let coordinate_bounds_id = self.write_coordinate_bounds(
            coordinates.type_id,
            image_id,
            level_id,
            selection.block(),
        );

        // Compare the coordinates against the bounds.
        let coords_numeric_type = match coordinates.size {
            Some(size) => NumericType::Vector {
                size,
                scalar: crate::Scalar::BOOL,
            },
            None => NumericType::Scalar(crate::Scalar::BOOL),
        };
        let coords_bool_type_id = self.get_numeric_type_id(coords_numeric_type);
        let coords_conds_id = self.gen_id();
        selection.block().body.push(Instruction::binary(
            spirv::Op::ULessThan,
            coords_bool_type_id,
            coords_conds_id,
            coordinates.value_id,
            coordinate_bounds_id,
        ));

        // If the comparison above was a vector comparison, then we need to
        // check that all components of the comparison are true.
        let coords_cond_id = if coords_bool_type_id != bool_type_id {
            let id = self.gen_id();
            selection.block().body.push(Instruction::relational(
                spirv::Op::All,
                bool_type_id,
                id,
                coords_conds_id,
            ));
            id
        } else {
            coords_conds_id
        };

        selection.if_true(self, coords_cond_id, null_id);

        // All conditions are met. We can carry out the access.
        let texel_id = access.generate(
            &mut self.writer.id_gen,
            coordinates.value_id,
            level_id,
            sample_id,
            selection.block(),
        );

        // This, then, is the value of the 'true' branch.
        Ok(selection.finish(self, texel_id))
    }

    /// Generate code for an `ImageLoad` expression.
    ///
    /// The arguments are the components of an `Expression::ImageLoad` variant.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_image_load(
        &mut self,
        result_type_id: Word,
        image: Handle<crate::Expression>,
        coordinate: Handle<crate::Expression>,
        array_index: Option<Handle<crate::Expression>>,
        level: Option<Handle<crate::Expression>>,
        sample: Option<Handle<crate::Expression>>,
        block: &mut Block,
    ) -> Result<Word, Error> {
        let image_type = self.fun_info[image].ty.inner_with(&self.ir_module.types);
        let image_class = match *image_type {
            crate::TypeInner::Image { class, .. } => class,
            _ => return Err(Error::Validation("image type")),
        };

        if image_class == crate::ImageClass::External {
            let coordinate_ty = self.fun_info[coordinate].ty.inner_with(&self.ir_module.types);
            let coord_id = if let &crate::TypeInner::Vector { scalar: crate::Scalar::I32, .. } = coordinate_ty {
                let coord_id = self.gen_id();
                block.body.push(Instruction::unary(
                    spirv::Op::Bitcast,
                    self.get_numeric_type_id(NumericType::Vector {
                        size: crate::VectorSize::Bi,
                        scalar: crate::Scalar::U32,
                    }),
                    coord_id,
                    self.cached[coordinate]
                ));
                coord_id
            } else {
                self.cached[coordinate]
            };
            let result_id = self.gen_id();
            let vec4f_type_id = self.get_numeric_type_id(NumericType::Vector {
                size: crate::VectorSize::Quad,
                scalar: crate::Scalar::F32,
            });
            let function_id = *self
                .writer
                .wrapped_functions
                .get(&WrappedFunction::ExternalTextureLoad)
                .unwrap();
            block.body.push(Instruction::function_call(
                vec4f_type_id,
                result_id,
                function_id,
                &[
                    self.get_external_texture_plane_id(image, 0),
                    self.get_external_texture_plane_id(image, 1),
                    self.get_external_texture_plane_id(image, 2),
                    self.get_external_texture_params_id(image),
                    coord_id,
                ],
            ));
            return Ok(result_id);
        }

        let image_id = self.get_handle_id(image);
        let access = Load::from_image_expr(self, image_id, image_class, result_type_id)?;
        let coordinates = self.write_image_coordinates(coordinate, array_index, block)?;

        let level_id = level.map(|expr| self.cached[expr]);
        let sample_id = sample.map(|expr| self.cached[expr]);

        // Perform the access, according to the bounds check policy.
        let access_id = match self.writer.bounds_check_policies.image_load {
            crate::proc::BoundsCheckPolicy::Restrict => {
                let (coords, level_id, sample_id) = self.write_restricted_coordinates(
                    image_id,
                    coordinates,
                    level_id,
                    sample_id,
                    block,
                )?;
                access.generate(&mut self.writer.id_gen, coords, level_id, sample_id, block)
            }
            crate::proc::BoundsCheckPolicy::ReadZeroSkipWrite => self
                .write_conditional_image_access(
                    image_id,
                    coordinates,
                    level_id,
                    sample_id,
                    block,
                    &access,
                )?,
            crate::proc::BoundsCheckPolicy::Unchecked => access.generate(
                &mut self.writer.id_gen,
                coordinates.value_id,
                level_id,
                sample_id,
                block,
            ),
        };

        // For depth images, `ImageLoad` expressions produce a single f32,
        // whereas the SPIR-V instructions always produce a vec4. So we may have
        // to pull out the component we need.
        let result_id = if result_type_id == access.result_type() {
            // The instruction produced the type we expected. We can use
            // its result as-is.
            access_id
        } else {
            // For `ImageClass::Depth` images, SPIR-V gave us four components,
            // but we only want the first one.
            let component_id = self.gen_id();
            block.body.push(Instruction::composite_extract(
                result_type_id,
                component_id,
                access_id,
                &[0],
            ));
            component_id
        };

        Ok(result_id)
    }

    /// Generate code for an `ImageSample` expression.
    ///
    /// The arguments are the components of an `Expression::ImageSample` variant.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_image_sample(
        &mut self,
        result_type_id: Word,
        image: Handle<crate::Expression>,
        sampler: Handle<crate::Expression>,
        gather: Option<crate::SwizzleComponent>,
        coordinate: Handle<crate::Expression>,
        array_index: Option<Handle<crate::Expression>>,
        offset: Option<Handle<crate::Expression>>,
        level: crate::SampleLevel,
        depth_ref: Option<Handle<crate::Expression>>,
        clamp_to_edge: bool,
        block: &mut Block,
    ) -> Result<Word, Error> {
        let image_type = self.fun_info[image].ty.handle().unwrap();
        let image_ty_inner = &self.ir_module.types[image_type].inner;
        let sampler_id = self.get_handle_id(sampler);

        if image_ty_inner.is_external_image() {
            let result_id = self.gen_id();
            let vec4f_type_id = self.get_numeric_type_id(NumericType::Vector {
                size: crate::VectorSize::Quad,
                scalar: crate::Scalar::F32,
            });
            let function_id = *self
                .writer
                .wrapped_functions
                .get(&WrappedFunction::ExternalTextureSample)
                .unwrap();
            block.body.push(Instruction::function_call(
                vec4f_type_id,
                result_id,
                function_id,
                &[
                    sampler_id,
                    self.get_external_texture_plane_id(image, 0),
                    self.get_external_texture_plane_id(image, 1),
                    self.get_external_texture_plane_id(image, 2),
                    self.get_external_texture_params_id(image),
                    self.cached[coordinate],
                ],
            ));
            return Ok(result_id);
        }

        // SPIR-V doesn't know about our `Depth` class, and it returns
        // `vec4<f32>`, so we need to grab the first component out of it.
        let needs_sub_access = match *image_ty_inner {
            crate::TypeInner::Image {
                class: crate::ImageClass::Depth { .. },
                ..
            } => depth_ref.is_none() && gather.is_none(),
            _ => false,
        };
        let sample_result_type_id = if needs_sub_access {
            self.get_numeric_type_id(NumericType::Vector {
                size: crate::VectorSize::Quad,
                scalar: crate::Scalar::F32,
            })
        } else {
            result_type_id
        };

        let image_id = self.get_handle_id(image);
        // OpTypeSampledImage
        let image_type_id = self.get_handle_type_id(image_type);
        let sampled_image_type_id =
            self.get_type_id(LookupType::Local(LocalType::SampledImage { image_type_id }));

        let coordinates = self.write_image_coordinates(coordinate, array_index, block)?;
        let coordinates_id = if clamp_to_edge {
            self.writer.require_any(
                "clamp sample coordinates to edge",
                &[spirv::Capability::ImageQuery],
            )?;

            // clamp_to_edge can only be used with Level 0, and no array offset, offset,
            // depth_ref or gather. This should have been caught by validation. Rather
            // than entirely duplicate validation code here just ensure the level is
            // zero, as we rely on that to query the texture size in order to calculate
            // the clamped coordinates.
            if level != crate::SampleLevel::Zero {
                return Err(Error::Validation(
                    "ImageSample::clamp_to_edge requires SampleLevel::Zero",
                ));
            }

            // Query the size of level 0 of the texture.
            let image_size_id = self.gen_id();
            let vec2u_type_id = self.writer.get_vec2u_type_id();
            let const_zero_uint_id = self.writer.get_constant_scalar(crate::Literal::U32(0));
            let mut query_inst = Instruction::image_query(
                spirv::Op::ImageQuerySizeLod,
                vec2u_type_id,
                image_size_id,
                image_id,
            );
            query_inst.add_operand(const_zero_uint_id);
            block.body.push(query_inst);

            let image_size_f_id = self.gen_id();
            let vec2f_type_id = self.writer.get_vec2f_type_id();
            block.body.push(Instruction::unary(
                spirv::Op::ConvertUToF,
                vec2f_type_id,
                image_size_f_id,
                image_size_id,
            ));

            // Calculate the top-left and bottom-right margin for clamping to. I.e. a
            // half-texel from each side.
            let const_0_5_f32_id = self.writer.get_constant_scalar(crate::Literal::F32(0.5));
            let const_0_5_vec2f_id = self.writer.get_constant_composite(
                LookupType::Local(LocalType::Numeric(NumericType::Vector {
                    size: crate::VectorSize::Bi,
                    scalar: crate::Scalar::F32,
                })),
                &[const_0_5_f32_id, const_0_5_f32_id],
            );

            let margin_left_id = self.gen_id();
            block.body.push(Instruction::binary(
                spirv::Op::FDiv,
                vec2f_type_id,
                margin_left_id,
                const_0_5_vec2f_id,
                image_size_f_id,
            ));

            let const_1_f32_id = self.writer.get_constant_scalar(crate::Literal::F32(1.0));
            let const_1_vec2f_id = self.writer.get_constant_composite(
                LookupType::Local(LocalType::Numeric(NumericType::Vector {
                    size: crate::VectorSize::Bi,
                    scalar: crate::Scalar::F32,
                })),
                &[const_1_f32_id, const_1_f32_id],
            );

            let margin_right_id = self.gen_id();
            block.body.push(Instruction::binary(
                spirv::Op::FSub,
                vec2f_type_id,
                margin_right_id,
                const_1_vec2f_id,
                margin_left_id,
            ));

            // Clamp the coords to the calculated margins
            let clamped_coords_id = self.gen_id();
            block.body.push(Instruction::ext_inst_gl_op(
                self.writer.gl450_ext_inst_id,
                spirv::GlslStd450Op::NClamp,
                vec2f_type_id,
                clamped_coords_id,
                &[coordinates.value_id, margin_left_id, margin_right_id],
            ));

            clamped_coords_id
        } else {
            coordinates.value_id
        };

        let sampled_image_id = self.gen_id();
        block.body.push(Instruction::sampled_image(
            sampled_image_type_id,
            sampled_image_id,
            image_id,
            sampler_id,
        ));
        let id = self.gen_id();

        let depth_id = depth_ref.map(|handle| self.cached[handle]);
        let mut mask = spirv::ImageOperands::empty();
        mask.set(spirv::ImageOperands::CONST_OFFSET, offset.is_some());

        let mut main_instruction = match (level, gather) {
            (_, Some(component)) => {
                let component_id = self.get_index_constant(component as u32);
                let mut inst = Instruction::image_gather(
                    sample_result_type_id,
                    id,
                    sampled_image_id,
                    coordinates_id,
                    component_id,
                    depth_id,
                );
                if !mask.is_empty() {
                    inst.add_operand(mask.bits());
                }
                inst
            }
            (crate::SampleLevel::Zero, None) => {
                let mut inst = Instruction::image_sample(
                    sample_result_type_id,
                    id,
                    SampleLod::Explicit,
                    sampled_image_id,
                    coordinates_id,
                    depth_id,
                );

                let zero_id = self.writer.get_constant_scalar(crate::Literal::F32(0.0));

                mask |= spirv::ImageOperands::LOD;
                inst.add_operand(mask.bits());
                inst.add_operand(zero_id);

                inst
            }
            (crate::SampleLevel::Auto, None) => {
                let mut inst = Instruction::image_sample(
                    sample_result_type_id,
                    id,
                    SampleLod::Implicit,
                    sampled_image_id,
                    coordinates_id,
                    depth_id,
                );
                if !mask.is_empty() {
                    inst.add_operand(mask.bits());
                }
                inst
            }
            (crate::SampleLevel::Exact(lod_handle), None) => {
                let mut inst = Instruction::image_sample(
                    sample_result_type_id,
                    id,
                    SampleLod::Explicit,
                    sampled_image_id,
                    coordinates_id,
                    depth_id,
                );

                let mut lod_id = self.cached[lod_handle];
                // SPIR-V expects the LOD to be a float for all image classes.
                // lod_id, however, will be an integer for depth images,
                // therefore we must do a conversion.
                if matches!(
                    self.ir_module.types[image_type].inner,
                    crate::TypeInner::Image {
                        class: crate::ImageClass::Depth { .. },
                        ..
                    }
                ) {
                    let lod_f32_id = self.gen_id();
                    let f32_type_id =
                        self.get_numeric_type_id(NumericType::Scalar(crate::Scalar::F32));
                    let convert_op = match *self.fun_info[lod_handle]
                        .ty
                        .inner_with(&self.ir_module.types)
                    {
                        crate::TypeInner::Scalar(crate::Scalar {
                            kind: crate::ScalarKind::Uint,
                            width: 4,
                        }) => spirv::Op::ConvertUToF,
                        crate::TypeInner::Scalar(crate::Scalar {
                            kind: crate::ScalarKind::Sint,
                            width: 4,
                        }) => spirv::Op::ConvertSToF,
                        _ => unreachable!(),
                    };
                    block.body.push(Instruction::unary(
                        convert_op,
                        f32_type_id,
                        lod_f32_id,
                        lod_id,
                    ));
                    lod_id = lod_f32_id;
                }
                mask |= spirv::ImageOperands::LOD;
                inst.add_operand(mask.bits());
                inst.add_operand(lod_id);

                inst
            }
            (crate::SampleLevel::Bias(bias_handle), None) => {
                let mut inst = Instruction::image_sample(
                    sample_result_type_id,
                    id,
                    SampleLod::Implicit,
                    sampled_image_id,
                    coordinates_id,
                    depth_id,
                );

                let bias_id = self.cached[bias_handle];
                mask |= spirv::ImageOperands::BIAS;
                inst.add_operand(mask.bits());
                inst.add_operand(bias_id);

                inst
            }
            (crate::SampleLevel::Gradient { x, y }, None) => {
                let mut inst = Instruction::image_sample(
                    sample_result_type_id,
                    id,
                    SampleLod::Explicit,
                    sampled_image_id,
                    coordinates_id,
                    depth_id,
                );

                let x_id = self.cached[x];
                let y_id = self.cached[y];
                mask |= spirv::ImageOperands::GRAD;
                inst.add_operand(mask.bits());
                inst.add_operand(x_id);
                inst.add_operand(y_id);

                inst
            }
        };

        if let Some(offset_const) = offset {
            let offset_id = self.cached[offset_const];
            main_instruction.add_operand(offset_id);
        }

        block.body.push(main_instruction);

        let id = if needs_sub_access {
            let sub_id = self.gen_id();
            block.body.push(Instruction::composite_extract(
                result_type_id,
                sub_id,
                id,
                &[0],
            ));
            sub_id
        } else {
            id
        };

        Ok(id)
    }

    /// Generate code for an `ImageQuery` expression.
    ///
    /// The arguments are the components of an `Expression::ImageQuery` variant.
    pub(super) fn write_image_query(
        &mut self,
        result_type_id: Word,
        image: Handle<crate::Expression>,
        query: crate::ImageQuery,
        block: &mut Block,
    ) -> Result<Word, Error> {
        use crate::{ImageClass as Ic, ImageDimension as Id, ImageQuery as Iq};

        let image_type = self.fun_info[image].ty.handle().unwrap();
        let (dim, arrayed, class) = match self.ir_module.types[image_type].inner {
            crate::TypeInner::Image {
                dim,
                arrayed,
                class,
            } => (dim, arrayed, class),
            _ => {
                return Err(Error::Validation("image type"));
            }
        };

        self.writer
            .require_any("image queries", &[spirv::Capability::ImageQuery])?;

        let id = if class == crate::ImageClass::External {
            let crate::ImageQuery::Size { .. } = query else {
                return Err(Error::Validation("invalid external image query"));
            };
            let function_id = *self
                .writer
                .wrapped_functions
                .get(&WrappedFunction::ExternalTextureSize)
                .unwrap();
            let result_id = self.gen_id();
            block.body.push(Instruction::function_call(
                self.writer.get_vec2u_type_id(),
                result_id,
                function_id,
                &[
                    self.get_external_texture_plane_id(image, 0),
                    self.get_external_texture_params_id(image),
                ],
            ));
            result_id
        } else {
            let image_id = self.get_handle_id(image);
            match query {
                Iq::Size { level } => {
                    let dim_coords = match dim {
                        Id::D1 => 1,
                        Id::D2 | Id::Cube => 2,
                        Id::D3 => 3,
                    };
                    let array_coords = usize::from(arrayed);
                    let vector_size = match dim_coords + array_coords {
                        2 => Some(crate::VectorSize::Bi),
                        3 => Some(crate::VectorSize::Tri),
                        4 => Some(crate::VectorSize::Quad),
                        _ => None,
                    };
                    let vector_numeric_type = match vector_size {
                        Some(size) => NumericType::Vector {
                            size,
                            scalar: crate::Scalar::U32,
                        },
                        None => NumericType::Scalar(crate::Scalar::U32),
                    };

                    let extended_size_type_id = self.get_numeric_type_id(vector_numeric_type);

                    let (query_op, level_id) = match class {
                        Ic::Sampled { multi: true, .. }
                        | Ic::Depth { multi: true }
                        | Ic::Storage { .. } => (spirv::Op::ImageQuerySize, None),
                        _ => {
                            let level_id = match level {
                                Some(expr) => self.cached[expr],
                                None => self.get_index_constant(0),
                            };
                            (spirv::Op::ImageQuerySizeLod, Some(level_id))
                        }
                    };

                    // The ID of the vector returned by SPIR-V, which contains the dimensions
                    // as well as the layer count.
                    let id_extended = self.gen_id();
                    let mut inst = Instruction::image_query(
                        query_op,
                        extended_size_type_id,
                        id_extended,
                        image_id,
                    );
                    if let Some(expr_id) = level_id {
                        inst.add_operand(expr_id);
                    }
                    block.body.push(inst);

                    if result_type_id != extended_size_type_id {
                        let id = self.gen_id();
                        let components = match dim {
                            // always pick the first component, and duplicate it for all 3 dimensions
                            Id::Cube => &[0u32, 0][..],
                            _ => &[0u32, 1, 2, 3][..dim_coords],
                        };
                        block.body.push(Instruction::vector_shuffle(
                            result_type_id,
                            id,
                            id_extended,
                            id_extended,
                            components,
                        ));

                        id
                    } else {
                        id_extended
                    }
                }
                Iq::NumLevels => {
                    let query_id = self.gen_id();
                    block.body.push(Instruction::image_query(
                        spirv::Op::ImageQueryLevels,
                        result_type_id,
                        query_id,
                        image_id,
                    ));

                    query_id
                }
                Iq::NumLayers => {
                    let vec_size = match dim {
                        Id::D1 => crate::VectorSize::Bi,
                        Id::D2 | Id::Cube => crate::VectorSize::Tri,
                        Id::D3 => crate::VectorSize::Quad,
                    };
                    let extended_size_type_id = self.get_numeric_type_id(NumericType::Vector {
                        size: vec_size,
                        scalar: crate::Scalar::U32,
                    });
                    let id_extended = self.gen_id();
                    let mut inst = Instruction::image_query(
                        spirv::Op::ImageQuerySizeLod,
                        extended_size_type_id,
                        id_extended,
                        image_id,
                    );
                    inst.add_operand(self.get_index_constant(0));
                    block.body.push(inst);

                    let extract_id = self.gen_id();
                    block.body.push(Instruction::composite_extract(
                        result_type_id,
                        extract_id,
                        id_extended,
                        &[vec_size as u32 - 1],
                    ));

                    extract_id
                }
                Iq::NumSamples => {
                    let query_id = self.gen_id();
                    block.body.push(Instruction::image_query(
                        spirv::Op::ImageQuerySamples,
                        result_type_id,
                        query_id,
                        image_id,
                    ));

                    query_id
                }
            }
        };

        Ok(id)
    }

    pub(super) fn write_image_store(
        &mut self,
        image: Handle<crate::Expression>,
        coordinate: Handle<crate::Expression>,
        array_index: Option<Handle<crate::Expression>>,
        value: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<(), Error> {
        let image_id = self.get_handle_id(image);
        let coordinates = self.write_image_coordinates(coordinate, array_index, block)?;
        let value_id = self.cached[value];

        let write = Store { image_id, value_id };

        match *self.fun_info[image].ty.inner_with(&self.ir_module.types) {
            crate::TypeInner::Image {
                class:
                    crate::ImageClass::Storage {
                        format: crate::StorageFormat::Bgra8Unorm,
                        ..
                    },
                ..
            } => self.writer.require_any(
                "Bgra8Unorm storage write",
                &[spirv::Capability::StorageImageWriteWithoutFormat],
            )?,
            _ => {}
        }

        write.generate(
            &mut self.writer.id_gen,
            coordinates.value_id,
            None,
            None,
            block,
        );

        Ok(())
    }

    pub(super) fn write_image_atomic(
        &mut self,
        image: Handle<crate::Expression>,
        coordinate: Handle<crate::Expression>,
        array_index: Option<Handle<crate::Expression>>,
        fun: crate::AtomicFunction,
        value: Handle<crate::Expression>,
        block: &mut Block,
    ) -> Result<(), Error> {
        let image_id = match self.ir_function.originating_global(image) {
            Some(handle) => self.writer.global_variables[handle].var_id,
            _ => return Err(Error::Validation("Unexpected image type")),
        };
        let crate::TypeInner::Image { class, .. } =
            *self.fun_info[image].ty.inner_with(&self.ir_module.types)
        else {
            return Err(Error::Validation("Invalid image type"));
        };
        let crate::ImageClass::Storage { format, .. } = class else {
            return Err(Error::Validation("Invalid image class"));
        };
        let scalar = format.into();
        let scalar_type_id = self.get_numeric_type_id(NumericType::Scalar(scalar));
        let pointer_type_id = self.get_pointer_type_id(scalar_type_id, spirv::StorageClass::Image);
        let signed = scalar.kind == crate::ScalarKind::Sint;
        if scalar.width == 8 {
            self.writer
                .require_any("64 bit image atomics", &[spirv::Capability::Int64Atomics])?;
        }
        let pointer_id = self.gen_id();
        let coordinates = self.write_image_coordinates(coordinate, array_index, block)?;
        let sample_id = self.writer.get_constant_scalar(crate::Literal::U32(0));
        block.body.push(Instruction::image_texel_pointer(
            pointer_type_id,
            pointer_id,
            image_id,
            coordinates.value_id,
            sample_id,
        ));

        let op = match fun {
            crate::AtomicFunction::Add => spirv::Op::AtomicIAdd,
            crate::AtomicFunction::Subtract => spirv::Op::AtomicISub,
            crate::AtomicFunction::And => spirv::Op::AtomicAnd,
            crate::AtomicFunction::ExclusiveOr => spirv::Op::AtomicXor,
            crate::AtomicFunction::InclusiveOr => spirv::Op::AtomicOr,
            crate::AtomicFunction::Min if signed => spirv::Op::AtomicSMin,
            crate::AtomicFunction::Min => spirv::Op::AtomicUMin,
            crate::AtomicFunction::Max if signed => spirv::Op::AtomicSMax,
            crate::AtomicFunction::Max => spirv::Op::AtomicUMax,
            crate::AtomicFunction::Exchange { .. } => {
                return Err(Error::Validation("Exchange atomics are not supported yet"))
            }
        };
        let result_type_id = self.get_expression_type_id(&self.fun_info[value].ty);
        let id = self.gen_id();
        let space = crate::AddressSpace::Handle;
        let (semantics, scope) = space.to_spirv_semantics_and_scope();
        let scope_constant_id = self.get_scope_constant(scope as u32);
        let semantics_id = self.get_index_constant(semantics.bits());
        let value_id = self.cached[value];

        block.body.push(Instruction::image_atomic(
            op,
            result_type_id,
            id,
            pointer_id,
            scope_constant_id,
            semantics_id,
            value_id,
        ));

        Ok(())
    }
}

/// External texture transfer function parameters.
///
/// This form, produced by [`write_load_tf_params`], has each component splatted to a vec3.
struct TfParams {
    a_id: Word,
    b_id: Word,
    g_id: Word,
    k_id: Word,
}

impl super::Writer {
    /// Write the SPIR-V function that implements [`crate::ImageQuery::Size`] for external textures.
    ///
    /// The function takes two arguments: the first plane, and the parameters. If the parameters
    /// specify a size, that size is returned, otherwise, the size of the first plane is returned.
    /// In both cases the returned value is a vec2u.
    ///
    /// The caller of the generated function is responsible for translating the parameters from the
    /// std140-compat form.
    pub(super) fn write_external_texture_size_query(
        &mut self,
        ir_module: &crate::Module,
    ) -> Result<(), Error> {
        let wrapped = WrappedFunction::ExternalTextureSize;
        if self.wrapped_functions.contains_key(&wrapped) {
            return Ok(());
        }

        // Querying the size of the first plane requires the `ImageQuery`
        // capability, just like any other image size query.
        self.require_any("image queries", &[spirv::Capability::ImageQuery])?;

        let vec2u_type_id = self.get_vec2u_type_id();
        let plane_type_id = self.get_type_id(LookupType::Local(LocalType::EXTERNAL_TEXTURE_PLANE));
        let plane_pointer_type_id =
            self.get_pointer_type_id(plane_type_id, spirv::StorageClass::UniformConstant);
        let params_type_id =
            self.get_handle_type_id(ir_module.special_types.external_texture_params.unwrap());

        let (mut function, [plane_param_id, params_param_id]) = self
            .write_wrapped_function_signature(
                wrapped,
                "naga_external_texture_size_query",
                &[
                    (plane_pointer_type_id, "plane0"),
                    (params_type_id, "params"),
                ],
                vec2u_type_id,
            );

        let label_id = self.id_gen.next();
        let mut block = Block::new(label_id);

        // ```
        // return select(
        //    imageQuerySizeLod(plane0, 0),
        //    params.size,
        //    params.size != vec2u(0),
        // )
        // ```

        // Load `params.size`.
        let params_size_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            vec2u_type_id,
            params_size_id,
            params_param_id,
            &[ExternalTextureParameter::Size as u32],
        ));

        // Query the size of the first plane's texture.
        let plane_id = self.id_gen.next();
        block.body.push(Instruction::load(
            plane_type_id,
            plane_id,
            plane_param_id,
            None,
        ));
        let lod_id = self.get_index_constant(0);
        let plane_size_id = self.id_gen.next();
        let mut plane_size_inst = Instruction::image_query(
            spirv::Op::ImageQuerySizeLod,
            vec2u_type_id,
            plane_size_id,
            plane_id,
        );
        plane_size_inst.add_operand(lod_id);
        block.body.push(plane_size_inst);

        // `params.size != vec2u(0)`
        let vec_condition_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::INotEqual,
            self.get_vec2_bool_type_id(),
            vec_condition_id,
            params_size_id,
            self.get_constant_null(vec2u_type_id),
        ));

        let condition_id = self.id_gen.next();
        block.body.push(Instruction::unary(
            spirv::Op::Any,
            self.get_bool_type_id(),
            condition_id,
            vec_condition_id,
        ));

        let result_id = self.id_gen.next();
        block.body.push(Instruction::select(
            vec2u_type_id,
            result_id,
            condition_id,
            params_size_id,
            plane_size_id,
        ));

        function.consume(block, Instruction::return_value(result_id));
        function.to_words(&mut self.logical_layout.function_definitions);

        Ok(())
    }

    /// Write the SPIR-V function that evaluates [`ImageLoad`][`crate::Expression::ImageLoad`]
    /// expressions for external textures.
    ///
    /// The function takes five arguments: the three external texture planes, the external
    /// texture parameters, and the coordinates to load. It returns a vec4f color value.
    ///
    /// The caller of the generated function is responsible for translating the parameters from the
    /// std140-compat form.
    pub(super) fn write_external_texture_image_load(
        &mut self,
        ir_module: &crate::Module,
    ) -> Result<(), Error> {
        // We need to do a size query and possibly yuv conversion as part of the load.
        // Writing the size query function will also check for the image query capability.
        self.write_external_texture_size_query(ir_module)?;
        self.write_external_texture_yuv_to_rgb(ir_module)?;

        let wrapped = WrappedFunction::ExternalTextureLoad;
        if self.wrapped_functions.contains_key(&wrapped) {
            return Ok(());
        }

        let plane_type_id = self.get_type_id(LookupType::Local(LocalType::EXTERNAL_TEXTURE_PLANE));
        let plane_pointer_type_id =
            self.get_pointer_type_id(plane_type_id, spirv::StorageClass::UniformConstant);
        let params_type_id =
            self.get_handle_type_id(ir_module.special_types.external_texture_params.unwrap());
        let f32_type_id = self.get_numeric_type_id(NumericType::Scalar(crate::Scalar::F32));
        let u32_type_id = self.get_numeric_type_id(NumericType::Scalar(crate::Scalar::U32));
        let vec2u_type_id = self.get_vec2u_type_id();
        let vec2f_type_id = self.get_vec2f_type_id();
        let vec4f_type_id = self.get_vector_type_id(crate::VectorSize::Quad, crate::Scalar::F32);
        let mat2x3f_type_id = self.get_numeric_type_id(NumericType::Matrix {
            columns: crate::VectorSize::Tri,
            rows: crate::VectorSize::Bi,
            scalar: crate::Scalar::F32,
        });

        let (mut function, [plane0_id, plane1_id, plane2_id, params_id, coord_id]) = self
            .write_wrapped_function_signature(
                wrapped,
                "naga_external_texture_image_load",
                &[
                    (plane_pointer_type_id, "plane0"),
                    (plane_pointer_type_id, "plane1"),
                    (plane_pointer_type_id, "plane2"),
                    (params_type_id, "params"),
                    (vec2u_type_id, "coord"),
                ],
                vec4f_type_id,
            );

        let label_id = self.id_gen.next();
        let mut block = Block::new(label_id);

        // Load handles
        let plane0_image_id = self.id_gen.next();
        block.body.push(Instruction::load(
            plane_type_id,
            plane0_image_id,
            plane0_id,
            None,
        ));
        let plane1_image_id = self.id_gen.next();
        block.body.push(Instruction::load(
            plane_type_id,
            plane1_image_id,
            plane1_id,
            None,
        ));
        let plane2_image_id = self.id_gen.next();
        block.body.push(Instruction::load(
            plane_type_id,
            plane2_image_id,
            plane2_id,
            None,
        ));

        // Prepare constants
        let vec2u_one_id = self.id_gen.next();
        block.body.push(Instruction::composite_construct(
            vec2u_type_id,
            vec2u_one_id,
            &[
                self.get_constant_scalar(crate::Literal::U32(1)),
                self.get_constant_scalar(crate::Literal::U32(1)),
            ]
        ));

        // Query the size
        let plane0_size_id = self.id_gen.next();
        block.body.push(Instruction::function_call(
            vec2u_type_id,
            plane0_size_id,
            self.wrapped_functions[&WrappedFunction::ExternalTextureSize],
            &[plane0_id, params_id],
        ));

        let plane0_size_float_id = self.id_gen.next();
        block.body.push(Instruction::unary(
            spirv::Op::ConvertUToF,
            vec2f_type_id,
            plane0_size_float_id,
            plane0_size_id,
        ));

        // Clamp coordinate to size, convert to float, extract x/y
        let plane0_size_minus_1_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::ISub,
            vec2u_type_id,
            plane0_size_minus_1_id,
            plane0_size_id,
            vec2u_one_id,
        ));

        let clamp_coord_id = self.id_gen.next();
        block.body.push(Instruction::ext_inst_gl_op(
            self.gl450_ext_inst_id,
            spirv::GlslStd450Op::UMin,
            vec2u_type_id,
            clamp_coord_id,
            &[coord_id, plane0_size_minus_1_id],
        ));

        let clamp_coord_float_id = self.id_gen.next();
        block.body.push(Instruction::unary(
            spirv::Op::ConvertUToF,
            vec2f_type_id,
            clamp_coord_float_id,
            clamp_coord_id,
        ));

        let x_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            f32_type_id,
            x_id,
            clamp_coord_float_id,
            &[0],
        ));

        let y_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            f32_type_id,
            y_id,
            clamp_coord_float_id,
            &[1],
        ));

        // Apply load transform
        let load_transform_in_id = self.id_gen.next();
        block.body.push(Instruction::composite_construct(
            self.get_vector_type_id(crate::VectorSize::Tri, crate::Scalar::F32),
            load_transform_in_id,
            &[x_id, y_id, self.get_constant_scalar(crate::Literal::F32(1.0))]
        ));

        let load_transform_matrix_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            mat2x3f_type_id,
            load_transform_matrix_id,
            params_id,
            &[ExternalTextureParameter::LoadTransform as u32],
        ));

        let load_transform_out_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::MatrixTimesVector,
            self.get_vector_type_id(crate::VectorSize::Bi, crate::Scalar::F32),
            load_transform_out_id,
            load_transform_matrix_id,
            load_transform_in_id,
        ));

        let plane0_coord_float_id = self.id_gen.next();
        block.body.push(Instruction::ext_inst_gl_op(
            self.gl450_ext_inst_id,
            spirv::GlslStd450Op::Round,
            vec2f_type_id,
            plane0_coord_float_id,
            &[load_transform_out_id],
        ));

        let plane0_coord_id = self.id_gen.next();
        block.body.push(Instruction::unary(
            spirv::Op::ConvertFToU,
            vec2u_type_id,
            plane0_coord_id,
            plane0_coord_float_id,
        ));

        // Load `params.num_planes`.
        let num_planes_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            u32_type_id,
            num_planes_id,
            params_id,
            &[ExternalTextureParameter::NumPlanes as u32],
        ));

        let label_num_planes_1_id = self.id_gen.next();
        let label_num_planes_2_id = self.id_gen.next();
        let label_num_planes_3_id = self.id_gen.next();

        let label_num_planes_merge_id = self.id_gen.next();
        block.body.push(Instruction::selection_merge(
            label_num_planes_merge_id,
            spirv::SelectionControl::NONE,
        ));

        function.consume(
            block,
            Instruction::switch(
                num_planes_id,
                label_num_planes_1_id, // default
                &[
                    super::instructions::Case { value: 2, label_id: label_num_planes_2_id },
                    super::instructions::Case { value: 3, label_id: label_num_planes_3_id },
                ],
            ),
        );

        let compute_plane1_coord_id = |writer: &mut super::Writer, block: &mut Block| {
            let plane1_size_id = writer.id_gen.next();
            let mut plane1_size_inst = Instruction::image_query(
                spirv::Op::ImageQuerySizeLod,
                vec2u_type_id,
                plane1_size_id,
                plane1_image_id,
            );
            plane1_size_inst.add_operand(writer.get_index_constant(0)); // lod
            block.body.push(plane1_size_inst);

            let plane1_size_float_id = writer.id_gen.next();
            block.body.push(Instruction::unary(
                spirv::Op::ConvertUToF,
                vec2f_type_id,
                plane1_size_float_id,
                plane1_size_id,
            ));

            let coord_mul_id = writer.id_gen.next();
            block.body.push(Instruction::binary(
                spirv::Op::FMul,
                vec2f_type_id,
                coord_mul_id,
                plane0_coord_float_id,
                plane1_size_float_id,
            ));

            let coord_div_id = writer.id_gen.next();
            block.body.push(Instruction::binary(
                spirv::Op::FDiv,
                vec2f_type_id,
                coord_div_id,
                coord_mul_id,
                plane0_size_float_id,
            ));

            let plane1_coord_float_id = writer.id_gen.next();
            block.body.push(Instruction::ext_inst_gl_op(
                writer.gl450_ext_inst_id,
                spirv::GlslStd450Op::Round,
                vec2f_type_id,
                plane1_coord_float_id,
                &[coord_div_id],
            ));

            let plane1_coord_id = writer.id_gen.next();
            block.body.push(Instruction::unary(
                spirv::Op::ConvertFToU,
                vec2u_type_id,
                plane1_coord_id,
                plane1_coord_float_id,
            ));

            plane1_coord_id
        };

        // num_planes == 3
        let value_if_3_planes_id = {
            let mut block = Block::new(label_num_planes_3_id);

            let plane1_coord_id = compute_plane1_coord_id(self, &mut block);

            // y_vec = tex.plane0.read(plane0_coords)
            let y_vec_id =
                Load {
                    opcode: spirv::Op::ImageFetch,
                    type_id: vec4f_type_id,
                    image_id: plane0_image_id,
                }
                .generate(
                    &mut self.id_gen,
                    plane0_coord_id,
                    None,
                    None,
                    &mut block,
                );

            // y = y_vec.x
            let y_id = self.id_gen.next();
            block.body.push(Instruction::composite_extract(
                f32_type_id,
                y_id,
                y_vec_id,
                &[0],
            ));

            let plane2_size_id = self.id_gen.next();
            let mut plane2_size_inst = Instruction::image_query(
                spirv::Op::ImageQuerySizeLod,
                vec2u_type_id,
                plane2_size_id,
                plane2_image_id,
            );
            plane2_size_inst.add_operand(self.get_index_constant(0)); // lod
            block.body.push(plane2_size_inst);

            let plane2_size_float_id = self.id_gen.next();
            block.body.push(Instruction::unary(
                spirv::Op::ConvertUToF,
                vec2f_type_id,
                plane2_size_float_id,
                plane2_size_id,
            ));

            let coord_mul_id = self.id_gen.next();
            block.body.push(Instruction::binary(
                spirv::Op::FMul,
                vec2f_type_id,
                coord_mul_id,
                plane0_coord_float_id,
                plane2_size_float_id,
            ));

            let coord_div_id = self.id_gen.next();
            block.body.push(Instruction::binary(
                spirv::Op::FDiv,
                vec2f_type_id,
                coord_div_id,
                coord_mul_id,
                plane0_size_float_id,
            ));

            let plane2_coord_float_id = self.id_gen.next();
            block.body.push(Instruction::ext_inst_gl_op(
                self.gl450_ext_inst_id,
                spirv::GlslStd450Op::Round,
                vec2f_type_id,
                plane2_coord_float_id,
                &[coord_div_id],
            ));

            let plane2_coord_id = self.id_gen.next();
            block.body.push(Instruction::unary(
                spirv::Op::ConvertFToU,
                vec2u_type_id,
                plane2_coord_id,
                plane2_coord_float_id,
            ));

            let u_vec_id =
                Load {
                    opcode: spirv::Op::ImageFetch,
                    type_id: vec4f_type_id,
                    image_id: plane1_image_id,
                }
                .generate(
                    &mut self.id_gen,
                    plane1_coord_id,
                    None,
                    None,
                    &mut block,
                );

            let v_vec_id =
                Load {
                    opcode: spirv::Op::ImageFetch,
                    type_id: vec4f_type_id,
                    image_id: plane2_image_id,
                }
                .generate(
                    &mut self.id_gen,
                    plane2_coord_id,
                    None,
                    None,
                    &mut block,
                );

            let u_id = self.id_gen.next();
            block.body.push(Instruction::composite_extract(
                f32_type_id,
                u_id,
                u_vec_id,
                &[0],
            ));

            let v_id = self.id_gen.next();
            block.body.push(Instruction::composite_extract(
                f32_type_id,
                v_id,
                v_vec_id,
                &[0],
            ));

            // Color space conversion
            let value_if_3_planes_id = self.id_gen.next();
            block.body.push(Instruction::function_call(
                vec4f_type_id,
                value_if_3_planes_id,
                self.wrapped_functions[&WrappedFunction::ExternalTextureYuvToRgb],
                &[y_id, u_id, v_id, params_id],
            ));

            function.consume(block, Instruction::branch(label_num_planes_merge_id));

            value_if_3_planes_id
        };

        // num_planes == 2
        let value_if_2_planes_id = {
            let mut block = Block::new(label_num_planes_2_id);

            let plane1_coord_id = compute_plane1_coord_id(self, &mut block);

            // y_vec = tex.plane0.read(plane0_coords)
            let y_vec_id =
                Load {
                    opcode: spirv::Op::ImageFetch,
                    type_id: vec4f_type_id,
                    image_id: plane0_image_id,
                }
                .generate(
                    &mut self.id_gen,
                    plane0_coord_id,
                    None,
                    None,
                    &mut block,
                );

            // y = y_vec.x
            let y_id = self.id_gen.next();
            block.body.push(Instruction::composite_extract(
                f32_type_id,
                y_id,
                y_vec_id,
                &[0],
            ));

            // uv = tex.plane1.read(plane1_coords)
            let uv_id =
                Load {
                    opcode: spirv::Op::ImageFetch,
                    type_id: vec4f_type_id,
                    image_id: plane1_image_id,
                }
                .generate(
                    &mut self.id_gen,
                    plane1_coord_id,
                    None,
                    None,
                    &mut block,
                );

            // u = uv.x
            let u_id = self.id_gen.next();
            block.body.push(Instruction::composite_extract(
                f32_type_id,
                u_id,
                uv_id,
                &[0],
            ));

            // v = uv.y
            let v_id = self.id_gen.next();
            block.body.push(Instruction::composite_extract(
                f32_type_id,
                v_id,
                uv_id,
                &[1],
            ));

            // Color space conversion
            let value_if_2_planes_id = self.id_gen.next();
            block.body.push(Instruction::function_call(
                vec4f_type_id,
                value_if_2_planes_id,
                self.wrapped_functions[&WrappedFunction::ExternalTextureYuvToRgb],
                &[y_id, u_id, v_id, params_id],
            ));

            function.consume(block, Instruction::branch(label_num_planes_merge_id));

            value_if_2_planes_id
        };

        // num_planes == 1
        let mut block = Block::new(label_num_planes_1_id);
        let value_if_1_plane_id =
            Load {
                opcode: spirv::Op::ImageFetch,
                type_id: vec4f_type_id,
                image_id: plane0_image_id,
            }
            .generate(
                &mut self.id_gen,
                plane0_coord_id,
                None,
                None,
                &mut block,
            );
        function.consume(block, Instruction::branch(label_num_planes_merge_id));

        // select
        let merged_value = self.id_gen.next();

        let mut block = Block::new(label_num_planes_merge_id);
        block.body.push(Instruction::phi(vec4f_type_id, merged_value,
            &[
                (value_if_1_plane_id, label_num_planes_1_id),
                (value_if_2_planes_id, label_num_planes_2_id),
                (value_if_3_planes_id, label_num_planes_3_id),
            ],
        ));

        function.consume(block, Instruction::return_value(merged_value));
        function.to_words(&mut self.logical_layout.function_definitions);

        Ok(())
    }

    /// Write the SPIR-V function that evaluates [`ImageSample`][`crate::Expression::ImageSample`]
    /// expressions for external textures.
    ///
    /// The function takes six arguments: a sampler, the three external texture planes, the
    /// external texture parameters, and the coordinates to sample. It returns a vec4f color value.
    ///
    /// The caller of the generated function is responsible for translating the parameters from the
    /// std140-compat form.
    pub(super) fn write_external_texture_image_sample(
        &mut self,
        ir_module: &crate::Module,
    ) -> Result<(), Error> {
        // We need to do a size query and possibly yuv conversion as part of the sample.
        // Writing the size query function will also check for the image query capability.
        self.write_external_texture_size_query(ir_module)?;
        self.write_external_texture_yuv_to_rgb(ir_module)?;

        let wrapped = WrappedFunction::ExternalTextureSample;
        if self.wrapped_functions.contains_key(&wrapped) {
            return Ok(());
        }

        let sampler_type_id = self.get_type_id(LookupType::Local(LocalType::Sampler));
        let plane_type_id = self.get_type_id(LookupType::Local(LocalType::EXTERNAL_TEXTURE_PLANE));
        let plane_sampled_type_id =
            self.get_type_id(LookupType::Local(LocalType::SampledImage { image_type_id: plane_type_id }));
        let plane_pointer_type_id =
            self.get_pointer_type_id(plane_type_id, spirv::StorageClass::UniformConstant);
        let params_type_id =
            self.get_handle_type_id(ir_module.special_types.external_texture_params.unwrap());
        let f32_type_id = self.get_numeric_type_id(NumericType::Scalar(crate::Scalar::F32));
        let u32_type_id = self.get_numeric_type_id(NumericType::Scalar(crate::Scalar::U32));
        let vec2u_type_id = self.get_vec2u_type_id();
        let vec2f_type_id = self.get_vec2f_type_id();
        let vec3f_type_id = self.get_vector_type_id(crate::VectorSize::Tri, crate::Scalar::F32);
        let vec4f_type_id = self.get_vector_type_id(crate::VectorSize::Quad, crate::Scalar::F32);
        let mat2x3f_type_id = self.get_numeric_type_id(NumericType::Matrix {
            columns: crate::VectorSize::Tri,
            rows: crate::VectorSize::Bi,
            scalar: crate::Scalar::F32,
        });

        let (mut function, [sampler_id, plane0_id, plane1_id, plane2_id, params_id, coord_id]) = self
            .write_wrapped_function_signature(
                wrapped,
                "naga_external_texture_image_sample",
                &[
                    (sampler_type_id, "sampler"),
                    (plane_pointer_type_id, "plane0"),
                    (plane_pointer_type_id, "plane1"),
                    (plane_pointer_type_id, "plane2"),
                    (params_type_id, "params"),
                    (vec2f_type_id, "coord"),
                ],
                vec4f_type_id,
            );

        let label_id = self.id_gen.next();
        let mut block = Block::new(label_id);

        // Load handles
        let plane0_image_id = self.id_gen.next();
        block.body.push(Instruction::load(
            plane_type_id,
            plane0_image_id,
            plane0_id,
            None,
        ));
        let plane1_image_id = self.id_gen.next();
        block.body.push(Instruction::load(
            plane_type_id,
            plane1_image_id,
            plane1_id,
            None,
        ));
        let plane2_image_id = self.id_gen.next();
        block.body.push(Instruction::load(
            plane_type_id,
            plane2_image_id,
            plane2_id,
            None,
        ));

        // Prepare constants
        let bounds_min_norm_id = self.id_gen.next();
        block.body.push(Instruction::composite_construct(
            vec3f_type_id,
            bounds_min_norm_id,
            &[
                self.get_constant_scalar(crate::Literal::F32(0.0)),
                self.get_constant_scalar(crate::Literal::F32(0.0)),
                self.get_constant_scalar(crate::Literal::F32(1.0)),
            ]
        ));

        let bounds_max_norm_id = self.id_gen.next();
        block.body.push(Instruction::composite_construct(
            vec3f_type_id,
            bounds_max_norm_id,
            &[
                self.get_constant_scalar(crate::Literal::F32(1.0)),
                self.get_constant_scalar(crate::Literal::F32(1.0)),
                self.get_constant_scalar(crate::Literal::F32(1.0)),
            ]
        ));

        let vec2f_0p5_id = self.id_gen.next();
        block.body.push(Instruction::composite_construct(
            vec2f_type_id,
            vec2f_0p5_id,
            &[
                self.get_constant_scalar(crate::Literal::F32(0.5)),
                self.get_constant_scalar(crate::Literal::F32(0.5)),
            ]
        ));

        // Calculate bounds. (params.size is irrelevant here as we are dealing with
        // normalized coordinates.)
        let sample_transform_matrix_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            mat2x3f_type_id,
            sample_transform_matrix_id,
            params_id,
            &[ExternalTextureParameter::SampleTransform as u32],
        ));

        let bounds_min_xform_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::MatrixTimesVector,
            vec2f_type_id,
            bounds_min_xform_id,
            sample_transform_matrix_id,
            bounds_min_norm_id,
        ));

        let bounds_max_xform_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::MatrixTimesVector,
            vec2f_type_id,
            bounds_max_xform_id,
            sample_transform_matrix_id,
            bounds_max_norm_id,
        ));

        let bounds_min_id = self.id_gen.next();
        block.body.push(Instruction::ext_inst_gl_op(
            self.gl450_ext_inst_id,
            spirv::GlslStd450Op::FMin,
            vec2f_type_id,
            bounds_min_id,
            &[bounds_min_xform_id, bounds_max_xform_id],
        ));

        let bounds_max_id = self.id_gen.next();
        block.body.push(Instruction::ext_inst_gl_op(
            self.gl450_ext_inst_id,
            spirv::GlslStd450Op::FMax,
            vec2f_type_id,
            bounds_max_id,
            &[bounds_min_xform_id, bounds_max_xform_id],
        ));

        // Extract coordinates. These are u/v normalized texture coordinates, but call them
        // x/y to avoid confusion with yuv texel data.
        let x_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            f32_type_id,
            x_id,
            coord_id,
            &[0],
        ));

        let y_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            f32_type_id,
            y_id,
            coord_id,
            &[1],
        ));

        let plane0_size_int_id = self.id_gen.next();
        let mut plane0_size_inst = Instruction::image_query(
            spirv::Op::ImageQuerySizeLod,
            vec2u_type_id,
            plane0_size_int_id,
            plane0_image_id,
        );
        plane0_size_inst.add_operand(self.get_index_constant(0)); // lod
        block.body.push(plane0_size_inst);

        let plane0_size_id = self.id_gen.next();
        block.body.push(Instruction::unary(
            spirv::Op::ConvertUToF,
            vec2f_type_id,
            plane0_size_id,
            plane0_size_int_id,
        ));

        // Apply sample transform
        let sample_transform_in_id = self.id_gen.next();
        block.body.push(Instruction::composite_construct(
            vec3f_type_id,
            sample_transform_in_id,
            &[x_id, y_id, self.get_constant_scalar(crate::Literal::F32(1.0))]
        ));

        let coord_transform_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::MatrixTimesVector,
            vec2f_type_id,
            coord_transform_id,
            sample_transform_matrix_id,
            sample_transform_in_id,
        ));

        let plane0_half_texel_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::FDiv,
            vec2f_type_id,
            plane0_half_texel_id,
            vec2f_0p5_id,
            plane0_size_id,
        ));

        let plane0_clamp_min_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::FAdd,
            vec2f_type_id,
            plane0_clamp_min_id,
            bounds_min_id,
            plane0_half_texel_id,
        ));

        let plane0_clamp_max_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::FSub,
            vec2f_type_id,
            plane0_clamp_max_id,
            bounds_max_id,
            plane0_half_texel_id,
        ));

        let plane0_coord_id = self.id_gen.next();
        block.body.push(Instruction::ext_inst_gl_op(
            self.gl450_ext_inst_id,
            spirv::GlslStd450Op::FClamp,
            vec2f_type_id,
            plane0_coord_id,
            &[coord_transform_id, plane0_clamp_min_id, plane0_clamp_max_id],
        ));

        // Load `params.num_planes`.
        let num_planes_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            u32_type_id,
            num_planes_id,
            params_id,
            &[ExternalTextureParameter::NumPlanes as u32],
        ));

        let label_num_planes_1_id = self.id_gen.next();
        let label_num_planes_2_id = self.id_gen.next();
        let label_num_planes_3_id = self.id_gen.next();

        let label_num_planes_merge_id = self.id_gen.next();
        block.body.push(Instruction::selection_merge(
            label_num_planes_merge_id,
            spirv::SelectionControl::NONE,
        ));

        function.consume(
            block,
            Instruction::switch(
                num_planes_id,
                label_num_planes_1_id, // default
                &[
                    super::instructions::Case { value: 2, label_id: label_num_planes_2_id },
                    super::instructions::Case { value: 3, label_id: label_num_planes_3_id },
                ],
            ),
        );

        let compute_plane1_coord_id = |writer: &mut super::Writer, block: &mut Block| {
            let plane1_size_int_id = writer.id_gen.next();
            let mut plane1_size_inst = Instruction::image_query(
                spirv::Op::ImageQuerySizeLod,
                vec2u_type_id,
                plane1_size_int_id,
                plane1_image_id,
            );
            plane1_size_inst.add_operand(writer.get_index_constant(0)); // lod
            block.body.push(plane1_size_inst);

            let plane1_size_id = writer.id_gen.next();
            block.body.push(Instruction::unary(
                spirv::Op::ConvertUToF,
                vec2f_type_id,
                plane1_size_id,
                plane1_size_int_id,
            ));

            let plane1_half_texel_id = writer.id_gen.next();
            block.body.push(Instruction::binary(
                spirv::Op::FDiv,
                vec2f_type_id,
                plane1_half_texel_id,
                vec2f_0p5_id,
                plane1_size_id,
            ));

            let plane1_clamp_min_id = writer.id_gen.next();
            block.body.push(Instruction::binary(
                spirv::Op::FAdd,
                vec2f_type_id,
                plane1_clamp_min_id,
                bounds_min_id,
                plane1_half_texel_id,
            ));

            let plane1_clamp_max_id = writer.id_gen.next();
            block.body.push(Instruction::binary(
                spirv::Op::FSub,
                vec2f_type_id,
                plane1_clamp_max_id,
                bounds_max_id,
                plane1_half_texel_id,
            ));

            let plane1_coord_id = writer.id_gen.next();
            block.body.push(Instruction::ext_inst_gl_op(
                writer.gl450_ext_inst_id,
                spirv::GlslStd450Op::FClamp,
                vec2f_type_id,
                plane1_coord_id,
                &[coord_transform_id, plane1_clamp_min_id, plane1_clamp_max_id],
            ));

            plane1_coord_id
        };

        // num_planes == 3
        let value_if_3_planes_id = {
            let mut block = Block::new(label_num_planes_3_id);

            let plane1_coord_id = compute_plane1_coord_id(self, &mut block);

            let plane0_sampled_id = self.id_gen.next();
            block.body.push(Instruction::sampled_image(
                plane_sampled_type_id,
                plane0_sampled_id,
                plane0_image_id,
                sampler_id,
            ));

            // y_vec = tex.plane0.read(plane0_coords)
            let y_vec_id = self.id_gen.next();
            let mut inst = Instruction::image_sample(
                vec4f_type_id,
                y_vec_id,
                SampleLod::Explicit,
                plane0_sampled_id,
                plane0_coord_id,
                None,
            );
            inst.add_operand(spirv::ImageOperands::LOD.bits());
            inst.add_operand(self.get_constant_scalar(crate::Literal::F32(0.0)));
            block.body.push(inst);

            // y = y_vec.x
            let y_id = self.id_gen.next();
            block.body.push(Instruction::composite_extract(
                f32_type_id,
                y_id,
                y_vec_id,
                &[0],
            ));

            let plane2_size_int_id = self.id_gen.next();
            let mut plane2_size_inst = Instruction::image_query(
                spirv::Op::ImageQuerySizeLod,
                vec2u_type_id,
                plane2_size_int_id,
                plane2_image_id,
            );
            plane2_size_inst.add_operand(self.get_index_constant(0)); // lod
            block.body.push(plane2_size_inst);

            let plane2_size_id = self.id_gen.next();
            block.body.push(Instruction::unary(
                spirv::Op::ConvertUToF,
                vec2f_type_id,
                plane2_size_id,
                plane2_size_int_id,
            ));

            let plane2_half_texel_id = self.id_gen.next();
            block.body.push(Instruction::binary(
                spirv::Op::FDiv,
                vec2f_type_id,
                plane2_half_texel_id,
                vec2f_0p5_id,
                plane2_size_id,
            ));

            let plane2_clamp_min_id = self.id_gen.next();
            block.body.push(Instruction::binary(
                spirv::Op::FAdd,
                vec2f_type_id,
                plane2_clamp_min_id,
                bounds_min_id,
                plane2_half_texel_id,
            ));

            let plane2_clamp_max_id = self.id_gen.next();
            block.body.push(Instruction::binary(
                spirv::Op::FSub,
                vec2f_type_id,
                plane2_clamp_max_id,
                bounds_max_id,
                plane2_half_texel_id,
            ));

            let plane2_coord_id = self.id_gen.next();
            block.body.push(Instruction::ext_inst_gl_op(
                self.gl450_ext_inst_id,
                spirv::GlslStd450Op::FClamp,
                vec2f_type_id,
                plane2_coord_id,
                &[coord_transform_id, plane2_clamp_min_id, plane2_clamp_max_id],
            ));

            let plane1_sampled_id = self.id_gen.next();
            block.body.push(Instruction::sampled_image(
                plane_sampled_type_id,
                plane1_sampled_id,
                plane1_image_id,
                sampler_id,
            ));

            let plane2_sampled_id = self.id_gen.next();
            block.body.push(Instruction::sampled_image(
                plane_sampled_type_id,
                plane2_sampled_id,
                plane2_image_id,
                sampler_id,
            ));

            let u_vec_id = self.id_gen.next();
            let mut inst = Instruction::image_sample(
                vec4f_type_id,
                u_vec_id,
                SampleLod::Explicit,
                plane1_sampled_id,
                plane1_coord_id,
                None,
            );
            inst.add_operand(spirv::ImageOperands::LOD.bits());
            inst.add_operand(self.get_constant_scalar(crate::Literal::F32(0.0)));
            block.body.push(inst);

            let v_vec_id = self.id_gen.next();
            let mut inst = Instruction::image_sample(
                vec4f_type_id,
                v_vec_id,
                SampleLod::Explicit,
                plane2_sampled_id,
                plane2_coord_id,
                None,
            );
            inst.add_operand(spirv::ImageOperands::LOD.bits());
            inst.add_operand(self.get_constant_scalar(crate::Literal::F32(0.0)));
            block.body.push(inst);

            let u_id = self.id_gen.next();
            block.body.push(Instruction::composite_extract(
                f32_type_id,
                u_id,
                u_vec_id,
                &[0],
            ));

            let v_id = self.id_gen.next();
            block.body.push(Instruction::composite_extract(
                f32_type_id,
                v_id,
                v_vec_id,
                &[0],
            ));

            // Color space conversion
            let value_if_3_planes_id = self.id_gen.next();
            block.body.push(Instruction::function_call(
                vec4f_type_id,
                value_if_3_planes_id,
                self.wrapped_functions[&WrappedFunction::ExternalTextureYuvToRgb],
                &[y_id, u_id, v_id, params_id],
            ));

            function.consume(block, Instruction::branch(label_num_planes_merge_id));

            value_if_3_planes_id
        };

        // num_planes == 2
        let value_if_2_planes_id = {
            let mut block = Block::new(label_num_planes_2_id);

            let plane1_coord_id = compute_plane1_coord_id(self, &mut block);

            let plane0_sampled_id = self.id_gen.next();
            block.body.push(Instruction::sampled_image(
                plane_sampled_type_id,
                plane0_sampled_id,
                plane0_image_id,
                sampler_id,
            ));

            let plane1_sampled_id = self.id_gen.next();
            block.body.push(Instruction::sampled_image(
                plane_sampled_type_id,
                plane1_sampled_id,
                plane1_image_id,
                sampler_id,
            ));

            // y_vec = tex.plane0.read(plane0_coords)
            let y_vec_id = self.id_gen.next();
            let mut inst = Instruction::image_sample(
                vec4f_type_id,
                y_vec_id,
                SampleLod::Explicit,
                plane0_sampled_id,
                plane0_coord_id,
                None,
            );
            inst.add_operand(spirv::ImageOperands::LOD.bits());
            inst.add_operand(self.get_constant_scalar(crate::Literal::F32(0.0)));
            block.body.push(inst);

            // y = y_vec.x
            let y_id = self.id_gen.next();
            block.body.push(Instruction::composite_extract(
                f32_type_id,
                y_id,
                y_vec_id,
                &[0],
            ));

            // uv = tex.plane1.read(plane1_coords)
            let uv_id = self.id_gen.next();
            let mut inst = Instruction::image_sample(
                vec4f_type_id,
                uv_id,
                SampleLod::Explicit,
                plane1_sampled_id,
                plane1_coord_id,
                None,
            );
            inst.add_operand(spirv::ImageOperands::LOD.bits());
            inst.add_operand(self.get_constant_scalar(crate::Literal::F32(0.0)));
            block.body.push(inst);

            // u = uv.x
            let u_id = self.id_gen.next();
            block.body.push(Instruction::composite_extract(
                f32_type_id,
                u_id,
                uv_id,
                &[0],
            ));

            // v = uv.y
            let v_id = self.id_gen.next();
            block.body.push(Instruction::composite_extract(
                f32_type_id,
                v_id,
                uv_id,
                &[1],
            ));

            // Color space conversion
            let value_if_2_planes_id = self.id_gen.next();
            block.body.push(Instruction::function_call(
                vec4f_type_id,
                value_if_2_planes_id,
                self.wrapped_functions[&WrappedFunction::ExternalTextureYuvToRgb],
                &[y_id, u_id, v_id, params_id],
            ));

            function.consume(block, Instruction::branch(label_num_planes_merge_id));

            value_if_2_planes_id
        };

        // num_planes == 1
        let value_if_1_plane_id = {
            let mut block = Block::new(label_num_planes_1_id);

            let plane0_sampled_id = self.id_gen.next();
            block.body.push(Instruction::sampled_image(
                plane_sampled_type_id,
                plane0_sampled_id,
                plane0_image_id,
                sampler_id,
            ));

            let value_if_1_plane_id = self.id_gen.next();
            let mut inst = Instruction::image_sample(
                vec4f_type_id,
                value_if_1_plane_id,
                SampleLod::Explicit,
                plane0_sampled_id,
                plane0_coord_id,
                None,
            );
            inst.add_operand(spirv::ImageOperands::LOD.bits());
            inst.add_operand(self.get_constant_scalar(crate::Literal::F32(0.0)));
            block.body.push(inst);
            function.consume(block, Instruction::branch(label_num_planes_merge_id));

            value_if_1_plane_id
        };

        // select
        let merged_value = self.id_gen.next();

        let mut block = Block::new(label_num_planes_merge_id);
        block.body.push(Instruction::phi(vec4f_type_id, merged_value,
            &[
                (value_if_1_plane_id, label_num_planes_1_id),
                (value_if_2_planes_id, label_num_planes_2_id),
                (value_if_3_planes_id, label_num_planes_3_id),
            ],
        ));

        function.consume(block, Instruction::return_value(merged_value));
        function.to_words(&mut self.logical_layout.function_definitions);

        Ok(())
    }

    fn write_load_tf_params(
        &mut self,
        ir_module: &crate::Module,
        block: &mut Block,
        params_id: Word,
        which: ExternalTextureParameter,
    ) -> TfParams {
        const TF_A: u32 = 0;
        const TF_B: u32 = 1;
        const TF_G: u32 = 2;
        const TF_K: u32 = 3;
        const _: () = {
            assert!(core::mem::offset_of!(nt::ExternalTextureTransferFunction, a) == TF_A as usize);
            assert!(
                core::mem::offset_of!(nt::ExternalTextureTransferFunction, b)
                    == TF_B as usize * size_of::<f32>()
            );
            assert!(
                core::mem::offset_of!(nt::ExternalTextureTransferFunction, g)
                    == TF_G as usize * size_of::<f32>()
            );
            assert!(
                core::mem::offset_of!(nt::ExternalTextureTransferFunction, k)
                    == TF_K as usize * size_of::<f32>()
            );
        };

        debug_assert!([
            ExternalTextureParameter::SrcTf,
            ExternalTextureParameter::DstTf
        ]
        .contains(&which));

        let vec3f_type_id = self.get_vector_type_id(crate::VectorSize::Tri, crate::Scalar::F32);
        let tf_type_id =
            self.get_handle_type_id(ir_module.special_types.external_texture_transfer_function.unwrap());

        let tf_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            tf_type_id,
            tf_id,
            params_id,
            &[which as u32],
        ));
        let a_scalar_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            self.get_f32_type_id(),
            a_scalar_id,
            tf_id,
            &[TF_A],
        ));
        let a_id = self.id_gen.next();
        block.body.push(Instruction::composite_construct(
            vec3f_type_id,
            a_id,
            &[a_scalar_id, a_scalar_id, a_scalar_id],
        ));
        let b_scalar_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            self.get_f32_type_id(),
            b_scalar_id,
            tf_id,
            &[TF_B],
        ));
        let b_id = self.id_gen.next();
        block.body.push(Instruction::composite_construct(
            vec3f_type_id,
            b_id,
            &[b_scalar_id, b_scalar_id, b_scalar_id],
        ));
        let g_scalar_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            self.get_f32_type_id(),
            g_scalar_id,
            tf_id,
            &[TF_G],
        ));
        let g_id = self.id_gen.next();
        block.body.push(Instruction::composite_construct(
            vec3f_type_id,
            g_id,
            &[g_scalar_id, g_scalar_id, g_scalar_id],
        ));
        let k_scalar_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            self.get_f32_type_id(),
            k_scalar_id,
            tf_id,
            &[TF_K],
        ));
        let k_id = self.id_gen.next();
        block.body.push(Instruction::composite_construct(
            vec3f_type_id,
            k_id,
            &[k_scalar_id, k_scalar_id, k_scalar_id],
        ));

        TfParams {
            a_id,
            b_id,
            g_id,
            k_id,
        }
    }

    /// Write the SPIR-V function that performs color space conversion for external texture
    /// [`ImageLoad`][`crate::Expression::ImageLoad`] and
    /// [`ImageSample`][`crate::Expression::ImageSample`] expressions.
    ///
    /// The function takes four arguments: the Y, U, and V values, and the external texture
    /// parameters. It returns a vec4f color value.
    ///
    /// TODO: This function is currently called only for 2- and 3-plane formats, but the
    /// logic related to gamut conversion and transfer functions should be applied to
    /// 1-plane formats as well. See <https://bugzilla.mozilla.org/show_bug.cgi?id=2071520>.
    fn write_external_texture_yuv_to_rgb(
        &mut self,
        ir_module: &crate::Module,
    ) -> Result<(), Error> {
        let wrapped = WrappedFunction::ExternalTextureYuvToRgb;
        if self.wrapped_functions.contains_key(&wrapped) {
            return Ok(());
        }

        let f32_type_id = self.get_f32_type_id();
        let vec3f_type_id = self.get_vector_type_id(crate::VectorSize::Tri, crate::Scalar::F32);
        let vec3_bool_type_id = self.get_vector_type_id(crate::VectorSize::Tri, crate::Scalar::BOOL);
        let vec4f_type_id = self.get_vector_type_id(crate::VectorSize::Quad, crate::Scalar::F32);
        let mat4x4f_type_id = self.get_numeric_type_id(NumericType::Matrix {
            columns: crate::VectorSize::Quad,
            rows: crate::VectorSize::Quad,
            scalar: crate::Scalar::F32,
        });
        let params_type_id =
            self.get_handle_type_id(ir_module.special_types.external_texture_params.unwrap());

        let (mut function, [y_id, u_id, v_id, params_id]) = self
            .write_wrapped_function_signature(
                wrapped,
                "naga_external_texture_yuv_to_rgb",
                &[
                    (f32_type_id, "y"),
                    (f32_type_id, "u"),
                    (f32_type_id, "v"),
                    (params_type_id, "params"),
                ],
                vec4f_type_id,
            );

        let label_id = self.id_gen.next();
        let mut block = Block::new(label_id);

        let vec3f_one_id = self.id_gen.next();
        block.body.push(Instruction::composite_construct(
            vec3f_type_id,
            vec3f_one_id,
            &[
                self.get_constant_scalar(crate::Literal::F32(1.0)),
                self.get_constant_scalar(crate::Literal::F32(1.0)),
                self.get_constant_scalar(crate::Literal::F32(1.0)),
            ]
        ));

        // Convert from YUV to non-linear RGB in the source color space.
        // `srcGammaRgba = params.yuv_conversion_matrix * vec4(y, u, v, 1.0)`
        let yuv_id = self.id_gen.next();
        block.body.push(Instruction::composite_construct(
            self.get_vector_type_id(crate::VectorSize::Quad, crate::Scalar::F32),
            yuv_id,
            &[
                y_id,
                u_id,
                v_id,
                self.get_constant_scalar(crate::Literal::F32(1.0)),
            ],
        ));

        let yuv_conversion_matrix_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            mat4x4f_type_id,
            yuv_conversion_matrix_id,
            params_id,
            &[ExternalTextureParameter::YuvConversionMatrix as u32],
        ));

        let src_gamma_rgba_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::MatrixTimesVector,
            self.get_vector_type_id(crate::VectorSize::Quad, crate::Scalar::F32),
            src_gamma_rgba_id,
            yuv_conversion_matrix_id,
            yuv_id,
        ));

        // `srcGammaRgb = srcGammaRgba.rgb`
        let src_gamma_rgb_id = self.id_gen.next();
        block.body.push(Instruction::vector_shuffle(
            vec3f_type_id,
            src_gamma_rgb_id,
            src_gamma_rgba_id,
            src_gamma_rgba_id,
            &[0, 1, 2],
        ));

        // Apply the inverse of the source transfer function to convert to
        // linear RGB in the source color space.
        // ```
        // srcLinearRgb = select(
        //     pow((srcGammaRgb + params.src_tf.a - 1.0) / params.src_tf.a, params.src_tf.g),
        //     srcGammaRgb / params.src_tf.k,
        //     srcGammaRgb < params.src_tf.k * params.src_tf.b,
        // );
        // ```
        let src_tf =
            self.write_load_tf_params(ir_module, &mut block, params_id, ExternalTextureParameter::SrcTf);

        // `srcGammaRgb < params.src_tf.k * params.src_tf.b`
        let src_k_times_b_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::FMul,
            vec3f_type_id,
            src_k_times_b_id,
            src_tf.k_id,
            src_tf.b_id,
        ));

        let src_tf_condition_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::FOrdLessThan,
            vec3_bool_type_id,
            src_tf_condition_id,
            src_gamma_rgb_id,
            src_k_times_b_id,
        ));

        // `srcGammaRgb + params.src_tf.a - 1.0`
        let src_tf_base_num1_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::FAdd,
            vec3f_type_id,
            src_tf_base_num1_id,
            src_gamma_rgb_id,
            src_tf.a_id,
        ));
        let src_tf_base_num_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::FSub,
            vec3f_type_id,
            src_tf_base_num_id,
            src_tf_base_num1_id,
            vec3f_one_id,
        ));

        // `pow((srcGammaRgb + params.src_tf.a - 1.0) / params.src_tf.a, params.src_tf.g)`
        let src_tf_base_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::FDiv,
            vec3f_type_id,
            src_tf_base_id,
            src_tf_base_num_id,
            src_tf.a_id,
        ));

        let src_tf_exponential_id = self.id_gen.next();
        block.body.push(Instruction::ext_inst_gl_op(
            self.gl450_ext_inst_id,
            spirv::GlslStd450Op::Pow,
            vec3f_type_id,
            src_tf_exponential_id,
            &[src_tf_base_id, src_tf.g_id],
        ));

        // `srcGammaRgb / params.src_tf.k`
        let src_tf_linear_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::FDiv,
            vec3f_type_id,
            src_tf_linear_id,
            src_gamma_rgb_id,
            src_tf.k_id,
        ));

        let src_linear_rgb_id = self.id_gen.next();
        block.body.push(Instruction::select(
            vec3f_type_id,
            src_linear_rgb_id,
            src_tf_condition_id,
            src_tf_linear_id,
            src_tf_exponential_id,
        ));

        // Multiply by the gamut conversion matrix to convert to linear RGB in
        // the destination color space.
        // `dstLinearRgb = mul(srcLinearRgb, {params}.gamut_conversion_matrix)`
        let gamut_conversion_matrix_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            self.get_numeric_type_id(NumericType::Matrix {
                columns: crate::VectorSize::Tri,
                rows: crate::VectorSize::Tri,
                scalar: crate::Scalar::F32,
            }),
            gamut_conversion_matrix_id,
            params_id,
            &[ExternalTextureParameter::GamutConversionMatrix as u32],
        ));

        let dst_linear_rgb_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::MatrixTimesVector,
            self.get_vector_type_id(crate::VectorSize::Tri, crate::Scalar::F32),
            dst_linear_rgb_id,
            gamut_conversion_matrix_id,
            src_linear_rgb_id,
        ));

        // Finally, apply the dest transfer function to convert to non-linear
        // RGB in the destination color space, and return the result.
        //
        // ```
        // dstGammaRgb = select(
        //     params.dst_tf.a * pow(dstLinearRgb, 1.0 / params.dst_tf.g) - (params.dst_tf.a - 1),
        //     params.dst_tf.k * dstLinearRgb,
        //     dstLinearRgb < params.dst_tf.b,
        // )
        // ```
        let dst_tf =
            self.write_load_tf_params(ir_module, &mut block, params_id, ExternalTextureParameter::DstTf);

        // `dstLinearRgb < params.src_tf.b`
        let dst_tf_condition_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::FOrdLessThan,
            vec3_bool_type_id,
            dst_tf_condition_id,
            dst_linear_rgb_id,
            dst_tf.b_id,
        ));

        let dst_tf_linear_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::FMul,
            vec3f_type_id,
            dst_tf_linear_id,
            dst_tf.k_id,
            dst_linear_rgb_id,
        ));

        // `params.dst_tf.a * pow(dstLinearRgb, 1.0 / params.dst_tf.g) - (params.dst_tf.a - 1)`
        let dst_tf_exponent_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::FDiv,
            vec3f_type_id,
            dst_tf_exponent_id,
            vec3f_one_id,
            dst_tf.g_id,
        ));

        let dst_tf_pow_id = self.id_gen.next();
        block.body.push(Instruction::ext_inst_gl_op(
            self.gl450_ext_inst_id,
            spirv::GlslStd450Op::Pow,
            vec3f_type_id,
            dst_tf_pow_id,
            &[dst_linear_rgb_id, dst_tf_exponent_id],
        ));

        let dst_tf_mul_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::FMul,
            vec3f_type_id,
            dst_tf_mul_id,
            dst_tf.a_id,
            dst_tf_pow_id,
        ));

        let dst_tf_a_minus_1_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::FSub,
            vec3f_type_id,
            dst_tf_a_minus_1_id,
            dst_tf.a_id,
            vec3f_one_id,
        ));

        let dst_tf_exponential_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::FSub,
            vec3f_type_id,
            dst_tf_exponential_id,
            dst_tf_mul_id,
            dst_tf_a_minus_1_id,
        ));

        let dst_gamma_rgb_id = self.id_gen.next();
        block.body.push(Instruction::select(
            vec3f_type_id,
            dst_gamma_rgb_id,
            dst_tf_condition_id,
            dst_tf_linear_id,
            dst_tf_exponential_id,
        ));

        // `dstGammaRgba = vec4(dstGammaRgb, 1.0)`
        let dst_gamma_rgba_id = self.id_gen.next();
        block.body.push(Instruction::vector_shuffle(
            vec4f_type_id,
            dst_gamma_rgba_id,
            dst_gamma_rgb_id,
            vec3f_one_id,
            &[0, 1, 2, 5],
        ));

        function.consume(block, Instruction::return_value(dst_gamma_rgba_id));
        function.to_words(&mut self.logical_layout.function_definitions);

        Ok(())
    }
}
