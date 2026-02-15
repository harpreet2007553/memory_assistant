import mongoose, { Schema } from "mongoose"

const chatSchema = new Schema(
    {
        user : {
            type : Schema.Types.ObjectId,
            ref : "User"
        },
        chats :{
                type : String,
        },
        SendBy : {
            type : String,
        }
    }
)

export const History = mongoose.model("History", chatSchema)