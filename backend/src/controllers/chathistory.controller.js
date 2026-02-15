import { History } from "../models/chatHistory.model";
import { asyncHandler } from "../utils/asyncHandler";
import { ApiError } from "../utils/apiError";
import { ApiResponse } from "../utils/apiResponse";

export const chatHistoryRetrieve = asyncHandler(async(req , res) => {
    const {username} = req.body;
    if(username === "" ){
        throw new ApiError(401, "Incomplete information is sent to the server")
    }

    const pageSize = 10; // number of chats per page
    const skip = (page - 1) * pageSize;

    userChats = await History.aggregate([
        {
            $match : {user : username},
        },
        {$lookup: {
        from: "User",              // name of User collection
        localField: "user",         // field in ChatHistory
        foreignField: "username",        // field in User
        as: "userChats"}},
        { $sort: { createdAt: -1 } },
        { $skip: skip },
        { $limit: pageSize },
        {$project : {
            chats : 1,
            SendBy : 1
        }}    
    ])
    res.json(
        {
            "chats" : userChats
        }
    )
})

export const chatStore = asyncHandler(async(req , res) => {
    const {username, chat, SendBy} = req.body;
    if(username === "" || chat === "" || SendBy === ""){
        throw new ApiError(401, "Incomplete information is sent to the server")
    }

    const newChat = await History.create(
        {
            username : username,
            chats : chat,
            SendBy : SendBy
        }
    )

    if(!newChat) throw new ApiError(500,"failed to create newChat")
    
    res.json(new ApiResponse(201, "chat stored successfully"))
})