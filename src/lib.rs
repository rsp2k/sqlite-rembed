// New lib.rs using genai - complete implementation
mod genai_client;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use genai_client::{EmbeddingClient, parse_client_options, legacy_provider_to_model};
use sqlite_loadable::{
    api, define_scalar_function, define_scalar_function_with_aux, define_virtual_table_writeablex,
    prelude::*, Error, Result,
};
use sqlite_loadable::table::{UpdateOperation, IndexInfo, VTab, VTabArguments, VTabCursor, VTabWriteable};
use sqlite_loadable::api::ValueType;
use sqlite_loadable::BestIndexError;
use std::{marker::PhantomData, mem, os::raw::c_int};
use zerocopy::AsBytes;
use base64;
use serde_json;

const FLOAT32_VECTOR_SUBTYPE: u8 = 223;
const CLIENT_OPTIONS_POINTER_NAME: &[u8] = b"sqlite-rembed-client-options\0";

pub fn rembed_version(context: *mut sqlite3_context, _values: &[*mut sqlite3_value]) -> Result<()> {
    api::result_text(context, format!("v{}-genai", env!("CARGO_PKG_VERSION")))?;
    Ok(())
}

pub fn rembed_debug(context: *mut sqlite3_context, _values: &[*mut sqlite3_value]) -> Result<()> {
    api::result_text(
        context,
        format!(
            "Version: v{}
Source: {}
Backend: genai v0.4.0-alpha.4
",
            env!("CARGO_PKG_VERSION"),
            env!("GIT_HASH")
        ),
    )?;
    Ok(())
}

pub fn rembed_client_options(
    context: *mut sqlite3_context,
    values: &[*mut sqlite3_value],
) -> Result<()> {
    if (values.len() % 2) != 0 {
        return Err(Error::new_message(
            "Must have an even number of arguments to rembed_client_options, as key/value pairs.",
        ));
    }

    let mut options: HashMap<String, String> = HashMap::new();
    let mut format: Option<String> = None;
    for pair in values.chunks(2) {
        let key = api::value_text(&pair[0])?;
        let value = api::value_text(&pair[1])?;
        if key == "format" {
            format = Some(value.to_owned());
        } else {
            options.insert(key.to_owned(), value.to_owned());
        }
    }

    // Build the model identifier based on format and options
    let model = if let Some(format) = format {
        // Legacy compatibility: convert old format to genai model
        let model_name = options.get("model")
            .ok_or_else(|| Error::new_message("'model' option is required"))?;
        legacy_provider_to_model(&format, model_name)
    } else if let Some(model) = options.get("model") {
        model.clone()
    } else {
        return Err(Error::new_message("'model' or 'format' key is required"));
    };

    let api_key = options.get("key").cloned()
        .or_else(|| options.get("api_key").cloned());

    // Create the client
    let client = EmbeddingClient::new(model, api_key)?;

    api::result_pointer(context, CLIENT_OPTIONS_POINTER_NAME, client);

    Ok(())
}

pub fn rembed(
    context: *mut sqlite3_context,
    values: &[*mut sqlite3_value],
    clients: &Rc<RefCell<HashMap<String, EmbeddingClient>>>,
) -> Result<()> {
    let client_name = api::value_text(&values[0])?;
    let input = api::value_text(&values[1])?;

    let clients_map = clients.borrow();
    let client = clients_map.get(client_name).ok_or_else(|| {
        Error::new_message(format!(
            "Client with name {} was not registered with rembed_clients.",
            client_name
        ))
    })?;

    // Generate embedding synchronously (blocks on async internally)
    let embedding = client.embed_sync(input)?;

    api::result_blob(context, embedding.as_bytes());
    api::result_subtype(context, FLOAT32_VECTOR_SUBTYPE);
    Ok(())
}

// Batch embedding function - accepts JSON array of texts
pub fn rembed_batch(
    context: *mut sqlite3_context,
    values: &[*mut sqlite3_value],
    clients: &Rc<RefCell<HashMap<String, EmbeddingClient>>>,
) -> Result<()> {
    let client_name = api::value_text(&values[0])?;
    let json_input = api::value_text(&values[1])?;

    // Parse JSON array of texts
    let texts: Vec<String> = serde_json::from_str(json_input)
        .map_err(|e| Error::new_message(format!("Invalid JSON array: {}", e)))?;

    if texts.is_empty() {
        return Err(Error::new_message("Input array cannot be empty"));
    }

    let clients_map = clients.borrow();
    let client = clients_map.get(client_name).ok_or_else(|| {
        Error::new_message(format!(
            "Client with name {} was not registered with rembed_clients.",
            client_name
        ))
    })?;

    // Generate embeddings in batch
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let embeddings = client.embed_batch_sync(text_refs)?;

    // Return as JSON array of base64-encoded embeddings
    let result: Vec<String> = embeddings.into_iter()
        .map(|embedding| {
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD.encode(embedding.as_bytes())
        })
        .collect();

    api::result_text(context, serde_json::to_string(&result)
        .map_err(|e| Error::new_message(format!("JSON serialization failed: {}", e)))?)?;
    Ok(())
}

// Virtual table implementation
enum Columns {
    Name,
    Options,
}

fn column(index: i32) -> Option<Columns> {
    match index {
        0 => Some(Columns::Name),
        1 => Some(Columns::Options),
        _ => None,
    }
}

#[repr(C)]
pub struct ClientsTable {
    base: sqlite3_vtab,
    clients: Rc<RefCell<HashMap<String, EmbeddingClient>>>,
}

impl<'vtab> VTab<'vtab> for ClientsTable {
    type Aux = Rc<RefCell<HashMap<String, EmbeddingClient>>>;
    type Cursor = ClientsCursor<'vtab>;

    fn create(
        db: *mut sqlite3,
        aux: Option<&Self::Aux>,
        args: VTabArguments,
    ) -> Result<(String, Self)> {
        Self::connect(db, aux, args)
    }

    fn connect(
        _db: *mut sqlite3,
        aux: Option<&Self::Aux>,
        _args: VTabArguments,
    ) -> Result<(String, ClientsTable)> {
        let base: sqlite3_vtab = unsafe { mem::zeroed() };
        let clients = aux.expect("Required aux").to_owned();

        let vtab = ClientsTable { base, clients };
        let sql = "create table x(name text primary key, options)".to_owned();

        Ok((sql, vtab))
    }

    fn destroy(&self) -> Result<()> {
        Ok(())
    }

    fn best_index(&self, mut info: IndexInfo) -> core::result::Result<(), BestIndexError> {
        info.set_estimated_cost(10000.0);
        info.set_estimated_rows(10000);
        info.set_idxnum(1);
        Ok(())
    }

    fn open(&'vtab mut self) -> Result<ClientsCursor<'vtab>> {
        ClientsCursor::new(self)
    }
}

impl<'vtab> VTabWriteable<'vtab> for ClientsTable {
    fn update(&'vtab mut self, operation: UpdateOperation<'_>, _p_rowid: *mut i64) -> Result<()> {
        match operation {
            UpdateOperation::Delete(_) => {
                return Err(Error::new_message(
                    "DELETE operations on rembed_clients is not supported yet",
                ))
            }
            UpdateOperation::Update { _values } => {
                return Err(Error::new_message(
                    "UPDATE operations on rembed_clients is not supported yet",
                ))
            }
            UpdateOperation::Insert { values, rowid: _ } => {
                let name = api::value_text(&values[0])?;

                let client = match api::value_type(&values[1]) {
                    ValueType::Text => {
                        let options = api::value_text(&values[1])?;
                        // Parse the options to get model and api key
                        let config = parse_client_options(name, options)?;
                        // Create client with the model and api key
                        EmbeddingClient::new(config.model, config.api_key)?
                    }
                    ValueType::Null => unsafe {
                        // Handle pointer from rembed_client_options
                        if let Some(client) =
                            api::value_pointer::<EmbeddingClient>(&values[1], CLIENT_OPTIONS_POINTER_NAME)
                        {
                            (*client).clone()
                        } else {
                            return Err(Error::new_message("client options required"));
                        }
                    },
                    _ => return Err(Error::new_message("client options required")),
                };

                self.clients.borrow_mut().insert(name.to_owned(), client);
            }
        }
        Ok(())
    }
}

#[repr(C)]
pub struct ClientsCursor<'vtab> {
    base: sqlite3_vtab_cursor,
    keys: Vec<String>,
    rowid: i64,
    phantom: PhantomData<&'vtab ClientsTable>,
}

impl ClientsCursor<'_> {
    fn new(table: &mut ClientsTable) -> Result<ClientsCursor> {
        let base: sqlite3_vtab_cursor = unsafe { mem::zeroed() };
        let c = table.clients.borrow();
        let keys = c.keys().map(|k| k.to_string()).collect();
        let cursor = ClientsCursor {
            base,
            keys,
            rowid: 0,
            phantom: PhantomData,
        };
        Ok(cursor)
    }
}

impl VTabCursor for ClientsCursor<'_> {
    fn filter(
        &mut self,
        _idx_num: c_int,
        _idx_str: Option<&str>,
        _values: &[*mut sqlite3_value],
    ) -> Result<()> {
        Ok(())
    }

    fn next(&mut self) -> Result<()> {
        self.rowid += 1;
        Ok(())
    }

    fn eof(&self) -> bool {
        (self.rowid as usize) >= self.keys.len()
    }

    fn column(&self, context: *mut sqlite3_context, i: c_int) -> Result<()> {
        let key = self
            .keys
            .get(self.rowid as usize)
            .expect("Internal rembed_clients logic error");
        match column(i) {
            Some(Columns::Name) => api::result_text(context, key)?,
            Some(Columns::Options) => (),
            None => (),
        };
        Ok(())
    }

    fn rowid(&self) -> Result<i64> {
        Ok(self.rowid)
    }
}

// For now, we'll focus on the scalar batch function approach
// Table function implementation can be added later when sqlite-loadable has better support

#[sqlite_entrypoint]
pub fn sqlite3_rembed_init(db: *mut sqlite3) -> Result<()> {
    let flags = FunctionFlags::UTF8
        | FunctionFlags::DETERMINISTIC
        | unsafe { FunctionFlags::from_bits_unchecked(0x001000000) };

    let clients: Rc<RefCell<HashMap<String, EmbeddingClient>>> =
        Rc::new(RefCell::new(HashMap::new()));

    define_scalar_function(
        db,
        "rembed_version",
        0,
        rembed_version,
        FunctionFlags::UTF8 | FunctionFlags::DETERMINISTIC,
    )?;

    define_scalar_function(
        db,
        "rembed_debug",
        0,
        rembed_debug,
        FunctionFlags::UTF8 | FunctionFlags::DETERMINISTIC,
    )?;

    define_scalar_function_with_aux(db, "rembed", 2, rembed, flags, Rc::clone(&clients))?;
    define_scalar_function_with_aux(db, "rembed", 3, rembed, flags, Rc::clone(&clients))?;

    define_scalar_function(
        db,
        "rembed_client_options",
        -1,
        rembed_client_options,
        flags,
    )?;

    define_virtual_table_writeablex::<ClientsTable>(db, "rembed_clients", Some(Rc::clone(&clients)))?;

    // Batch embedding function
    define_scalar_function_with_aux(
        db,
        "rembed_batch",
        2,
        rembed_batch,
        flags,
        Rc::clone(&clients),
    )?;

    // Table function will be added in a future version when sqlite-loadable has better support

    Ok(())
}