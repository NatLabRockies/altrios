use crate::imports::*;
use crate::utilities::TokenStreamIterator;

/// Name a field is serialized under -- the `#[serde(rename = "...")]` added by
/// `serde_api` for unit-bearing fields, or the bare ident otherwise.
///
/// This is the name the mask is keyed by from Python, so it has to match the
/// history vector's own key exactly or the two would drift apart.
fn serde_name(field: &syn::Field) -> String {
    let ident = field.ident.as_ref().unwrap().to_string();
    for attr in &field.attrs {
        if !attr.path().is_ident("serde") {
            continue;
        }
        let tokens = attr.to_token_stream().to_string();
        if let Some(pos) = tokens.find("rename") {
            let rest = &tokens[pos..];
            if let Some(open) = rest.find('"') {
                if let Some(len) = rest[open + 1..].find('"') {
                    return rest[open + 1..open + 1 + len].to_string();
                }
            }
        }
    }
    ident
}

/// True for the step counter, which is never maskable. See the mask struct doc.
fn is_index(ident: &syn::Ident) -> bool {
    *ident == "i"
}

pub(crate) fn history_vec_derive(input: TokenStream) -> TokenStream {
    let item_struct = syn::parse_macro_input!(input as syn::ItemStruct);
    let original_name = &item_struct.ident;
    let original_name_str: String = original_name.to_string();
    let new_name = Ident::new(
        &format!("{}HistoryVec", original_name.to_token_stream()),
        original_name.span(),
    );
    let mask_name = Ident::new(
        &format!("{}HistoryVecMask", original_name.to_token_stream()),
        original_name.span(),
    );
    let new_name_str: String = new_name.to_string();
    let fields = item_struct.fields;
    let field_names = fields
        .iter()
        .map(|f| f.ident.as_ref().unwrap())
        .collect::<Vec<_>>();

    // Fields split by whether they can be switched off. `i` cannot: every
    // history is required to be self-indexing, so that components recording at
    // different intervals can still be joined to each other and to the train.
    // Without it, a coarser component's samples could only be aligned by
    // position, which silently stops meaning anything the moment two intervals
    // differ. It also keeps `len` unambiguous -- see below.
    let maskable = field_names
        .iter()
        .copied()
        .filter(|id| !is_index(id))
        .collect::<Vec<_>>();
    let always = field_names
        .iter()
        .copied()
        .filter(|id| is_index(id))
        .collect::<Vec<_>>();
    let has_index = !always.is_empty();

    let maskable_serde_names = fields
        .iter()
        .filter(|f| !is_index(f.ident.as_ref().unwrap()))
        .map(serde_name)
        .collect::<Vec<_>>();

    let vec_fields = fields
        .iter()
        .map(|f| {
            let ident = f.ident.as_ref().unwrap();
            let ty = &f.ty;
            let attrs = &f.attrs.iter().collect::<Vec<&syn::Attribute>>();
            quote! {
                #(#attrs)*
                pub #ident: Vec<#ty>,
            }
        })
        .concat();

    // Only `rename` attributes carry over to the mask. A `#[serde(default)]`
    // copied from the state struct would mean a missing key deserializes to
    // `false` -- i.e. silently stops recording a field -- which is the exact
    // opposite of the intended default. Defaulting is handled at the container
    // level instead, off the hand-written `Default` impl below.
    let mask_fields = fields
        .iter()
        .filter(|f| !is_index(f.ident.as_ref().unwrap()))
        .map(|f| {
            let ident = f.ident.as_ref().unwrap();
            let renames = f
                .attrs
                .iter()
                .filter(|a| {
                    a.path().is_ident("serde")
                        && a.to_token_stream().to_string().contains("rename")
                })
                .collect::<Vec<&syn::Attribute>>();
            quote! {
                #(#renames)*
                pub #ident: bool,
            }
        })
        .concat();

    let vec_new = fields
        .iter()
        .map(|f| {
            let ident = f.ident.as_ref().unwrap();
            quote! {
                #ident: Vec::new(),
            }
        })
        .concat();

    // `i` is always recorded, so it is the length of the history whenever it
    // exists. Falling back to the longest vector covers the one `HistoryVec`
    // that has no step counter (`InitTrainState`), and is still correct when
    // every other field is masked off.
    let len_impl: TokenStream2 = if has_index {
        quote! { self.i.len() }
    } else {
        quote! {
            [ #(self.#field_names.len(),)* 0_usize ].into_iter().max().unwrap_or(0)
        }
    };

    let mut generated = TokenStream2::new();
    let struct_doc: TokenStream2 = format!("/// Stores history of {original_name_str}")
        .parse()
        .unwrap();
    let mask_doc: TokenStream2 = format!(
        "/// Which fields of [{new_name_str}] are recorded. Every field defaults to `true`; setting one to `false` leaves that vector empty and skips the clone, so unread series cost neither memory nor time. The step counter `i` has no entry here -- it is always recorded, so that histories taken at different `save_interval`s remain joinable."
    )
    .parse()
    .unwrap();
    let push_doc: TokenStream2 =
        format!("/// Pushes enabled fields of {original_name_str} to {new_name_str}")
            .parse()
            .unwrap();
    let pop_doc: TokenStream2 = format!(
        "/// Remove and return last element as {original_name_str}. Returns `None` if any field has been masked off, because the removed state could not be reconstructed in full."
    )
    .parse()
    .unwrap();
    let state_vec_doc: TokenStream2 = format!(
        "/// Return history as vec of {original_name_str}. Errors if any field has been masked off, rather than filling the gaps with defaults that would read as real zeroes downstream."
    )
    .parse()
    .unwrap();

    generated.append_all(quote! {
        #[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
        #[serde(default, deny_unknown_fields)]
        #mask_doc
        pub struct #mask_name {
            #mask_fields
        }

        impl Default for #mask_name {
            fn default() -> #mask_name {
                #mask_name {
                    #(#maskable: true,)*
                }
            }
        }

        impl #mask_name {
            /// True when nothing has been switched off
            pub fn is_all_enabled(&self) -> bool {
                #(self.#maskable &&)* true
            }

            /// Serialized names of the fields currently switched off
            pub fn disabled(&self) -> Vec<&'static str> {
                let mut names: Vec<&'static str> = Vec::new();
                #(
                    if !self.#maskable {
                        names.push(#maskable_serde_names);
                    }
                )*
                names
            }

            /// Switch every field off. Pair with re-enabling the few that are
            /// read, rather than listing the many that are not.
            pub fn disable_all(&mut self) {
                #(self.#maskable = false;)*
            }
        }

        impl Init for #mask_name { }
        impl SerdeAPI for #mask_name { }

        #[serde_api]
        #[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
        #[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq, from_py_object))]
        #struct_doc
        pub struct #new_name {
            #vec_fields
            /// Which fields above are recorded; all of them, by default.
            #[serde(default)]
            pub save_fields: #mask_name,
        }

        #[pyo3_api]
        impl #new_name {
            #[pyo3(name = "len")]
            fn len_py(&self) -> usize {
                self.len()
            }

            fn __len__(&self) -> usize {
                self.len()
            }
        }

        impl Init for #new_name { }
        impl SerdeAPI for #new_name { }

        impl #new_name {
            /// Creates new emtpy vec container
            pub fn new() -> #new_name {
                #new_name {
                    #vec_new
                    save_fields: #mask_name::default(),
                }
            }

            #push_doc
            pub fn push(&mut self, state: #original_name) {
                #(self.#always.push(state.#always.clone());)*
                #(
                    if self.save_fields.#maskable {
                        self.#maskable.push(state.#maskable.clone());
                    }
                )*
            }

            /// clear all history vecs, leaving the field mask as it is
            pub fn clear(&mut self) {
                #(self.#field_names.clear();)*
            }

            #pop_doc
            pub fn pop(&mut self) -> Option<#original_name> {
                if self.is_empty() || !self.save_fields.is_all_enabled() {
                    None
                } else {
                    #(
                        let #field_names = self.#field_names.pop().unwrap();
                    )*
                    Some(#original_name{#(#field_names: #field_names.clone()),*})
                }
            }

            /// Returns number of recorded time steps
            pub fn len(&self) -> usize {
                #len_impl
            }

            /// Returns True if no time steps have been recorded
            pub fn is_empty(&self) -> bool {
                self.len() == 0
            }

            #state_vec_doc
            pub fn state_vec(&self) -> anyhow::Result<Vec<#original_name>> {
                anyhow::ensure!(
                    self.save_fields.is_all_enabled(),
                    "`{}` cannot be reconstructed as states: {:?} were not recorded",
                    #new_name_str,
                    self.save_fields.disabled()
                );
                let mut state_vec: Vec<#original_name> = Vec::new();
                for i in 0..self.len() {
                    state_vec.push(
                        #original_name{
                            #(#field_names: self.#field_names[i].clone(),)*
                        }
                    )
                }
                Ok(state_vec)
            }

            // TODO: flesh this out
            // /// Returns fieldnames of any fields that are constant throughout history
            // pub fn names_of_static_fields(&self) -> Vec<String> {

            // }
        }

        impl Default for #new_name {
            fn default() -> #new_name {
                #new_name::new()
            }
        }
    });
    generated.into()
}
