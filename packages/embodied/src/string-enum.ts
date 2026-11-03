import { type TUnsafe, Type } from "typebox";

/**
 * A `{ type: "string", enum }` schema as a TypeBox type, so it composes with `Type.Object` and `Type.Optional`.
 *
 * `StringEnum` from `@earendil-works/pi-ai` returns a pi-schema `Schema` since pi 1.5.0 (TypeBox left pi-ai). Wrapping
 * that plain object in `Type.Optional` leaves an enumerable `~optional` key that is serialized into the tool schema sent
 * to providers, and TypeBox's `Static` types it as `unknown`; a `Type.Unsafe` keeps TypeBox's markers hidden and its
 * static type. This is the implementation pi-ai shipped before 1.5.0.
 */
export function StringEnum<T extends readonly string[]>(
	values: T,
	options?: { description?: string; default?: T[number] },
): TUnsafe<T[number]> {
	return Type.Unsafe<T[number]>({
		type: "string",
		enum: values as unknown as string[],
		...(options?.description && { description: options.description }),
		...(options?.default && { default: options.default }),
	});
}
