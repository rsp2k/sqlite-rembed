// New lib.rs using genai - rename to lib.rs when ready to switch
mod genai_client;
mod clients_vtab;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use genai_client::{EmbeddingClient, parse_client_options};
use clients_vtab::ClientsTable;
use sqlite_loadable::{
    api, define_scalar_function, define_scalar_function_with_aux, define_virtual_table_writeablex,
    prelude::*, Error, Result,
};
use zerocopy::AsBytes;

const FLOAT32_VECTOR_SUBTYPE: u8 = 223;
const CLIENT_OPTIONS_POINTER_NAME: &[u8] = b"sqlite-rembed-client-options\0";

pub fn rembed_version(context: *mut sqlite3_context, _values: &[*mut sqlite3_value]) -> Result<()> {
    api::result_text(context, format!("v{}", env!("CARGO_PKG_VERSION")))?;
    Ok(())
}

pub fn rembed_debug(context: *mut sqlite3_context, _values: &[*mut sqlite3_value]) -> Result<()> {
    api::result_text(
        context,
        format!(
            "Version: v{}
Source: {}
Provider: genai v0.4.0-alpha.4
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
    for pair in values.chunks(2) {
        let key = api::value_text(&pair[0])?;
        let value = api::value_text(&pair[1])?;
        options.insert(key.to_owned(), value.to_owned());
    }

    // For genai, we mainly need the model identifier
    // Format can be "provider::model" or just "model"
    let model = options.get("model")
        .or_else(|| options.get("format"))
        .ok_or_else(|| Error::new_message("'model' or 'format' key is required"))?
        .clone();

    let api_key = options.get("key").cloned();

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

    // Note: We need to update ClientsTable to work with EmbeddingClient
    // For now, commenting out to avoid compilation errors
    // define_virtual_table_writeablex::<ClientsTable>(db, "rembed_clients", Some(Rc::clone(&clients)))?;

    Ok(())
}